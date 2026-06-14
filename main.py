import sys
import queue
import threading
import numpy as np
import sounddevice as sd
import ctypes
import os
import io
from pathlib import Path
from PIL import ImageGrab 

from google import genai
from google.genai import types  
from groq import Groq  
import cohere  
from mistralai.client import Mistral  # Explicitly updated for modern v2.0.0+ structural paths
from PyQt6.QtCore import Qt, QPoint, QThread, pyqtSignal
from PyQt6.QtGui import QCursor, QKeyEvent
from PyQt6.QtWidgets import QApplication, QTextBrowser, QMainWindow, QVBoxLayout, QWidget, QSizeGrip, QGridLayout
from dotenv import load_dotenv

# --- SECURITY & ENVIRONMENT INITIALIZATION ---
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / '.env'
load_dotenv(dotenv_path=ENV_PATH)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

if not GEMINI_API_KEY:
    print(f"\n[ERROR] Missing GEMINI_API_KEY inside environment configuration.")
    sys.exit(1)

try:
    from transcriber import AudioTranscriber
    TRANSCRIBER_AVAILABLE = True
except ImportError:
    print("[WARNING] transcriber.py missing. Voice loop tracking disabled.")
    TRANSCRIBER_AVAILABLE = False

# Automatic silence detection config 
CHUNK_DURATION = 1.0     
SILENCE_THRESHOLD = 0.4   
SILENCE_CHUNKS_LIMIT = 2  

# Win32 Constants for invisible layering
WDA_EXCLUDEFROMCAPTURE = 0x00000011
GWL_EXSTYLE = -20
WS_EX_LAYERED = 0x80000
LWA_COLORKEY = 0x00000001

# --- SYSTEM PERSONA PROMPT CONFIGURATION ---
INTERVIEW_SYSTEM_INSTRUCTION = (
    "You are an expert technical advisor supporting a candidate during a live, elite Principal Software Engineer interview. "
    "Listen carefully to the question asked by the interviewer. Provide a highly precise, production-grade technical answer. "
    "CRITICAL SPECIFICATIONS:\n"
    "1. Do not include introductory text, greetings, fillers ('Sure, I can help with that'), or summaries.\n"
    "2. Start immediately with the answer body, architectural blueprint, or optimal code snippet.\n"
    "3. Keep explanations optimized for clear, spoken verbal delivery."
)


# --- COGNITIVE ENGINE THREAD WITH QUAD FAILOVER LAYER PROTECTION ---
class CopilotEngine(QThread):
    text_updated = pyqtSignal(str) 

    def __init__(self):
        super().__init__()
        self.audio_queue = queue.Queue()
        self.transcriber = AudioTranscriber() if TRANSCRIBER_AVAILABLE else None
        
        # Instantiate active clients safely
        self.ai_client = genai.Client(api_key=GEMINI_API_KEY)
        self.groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
        self.cohere_client = cohere.Client(api_key=COHERE_API_KEY) if COHERE_API_KEY else None
        self.mistral_client = Mistral(api_key=MISTRAL_API_KEY) if MISTRAL_API_KEY else None
        
        self.running = True
        self.conversation_history = []  # Strictly stores types.Content instances

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        self.audio_queue.put(indata.copy())

    def run(self):
        if not TRANSCRIBER_AVAILABLE:
            return

        # ISOLATION EXTRACTION: Target ONLY the isolated VB-Cable output device 
        devices = sd.query_devices()
        target_idx = None
        for idx, d in enumerate(devices):
            if "cable output" in d['name'].lower():
                target_idx = idx
                break
                
        if target_idx is None:
            print("[WARNING] VB-Cable not found. Defaulting to standard system input (microphone bleed possible).")
            target_idx = sd.query_devices(kind='input')['index']

        chosen_device = devices[target_idx]
        channels = min(2, chosen_device['max_input_channels'])
        sample_rate = int(chosen_device['default_samplerate'])
        chunk_samples = int(sample_rate * CHUNK_DURATION)

        speech_buffer = np.zeros((0, channels))
        consecutive_silence_count = 0
        has_spoken = False

        print(f"[SYSTEM] Listening via Isolated Device Target Index {target_idx} ({chosen_device['name']})")

        with sd.InputStream(device=target_idx, channels=channels, callback=self.audio_callback, samplerate=sample_rate):
            while self.running:
                try:
                    chunk = self.audio_queue.get(timeout=0.1)
                    speech_buffer = np.vstack((speech_buffer, chunk))
                except queue.Empty:
                    continue

                if len(speech_buffer) >= chunk_samples:
                    current_chunk = speech_buffer[-chunk_samples:]
                    mono_chunk = np.mean(current_chunk, axis=1) if channels == 2 else current_chunk.flatten()
                    volume = np.linalg.norm(mono_chunk)

                    if volume > SILENCE_THRESHOLD:
                        consecutive_silence_count = 0
                        has_spoken = True
                    else:
                        consecutive_silence_count += 1

                    if has_spoken and consecutive_silence_count >= SILENCE_CHUNKS_LIMIT:
                        mono_full_phrase = np.mean(speech_buffer, axis=1) if channels == 2 else speech_buffer.flatten()
                        speech_buffer = np.zeros((0, channels))
                        has_spoken = False
                        consecutive_silence_count = 0

                        captured_text = self.transcriber.transcribe_buffer(mono_full_phrase, sample_rate=sample_rate)
                        if captured_text.strip():
                            print(f"\n[Interviewer Query]: {captured_text}")
                            threading.Thread(target=self.process_interview_flow, args=(captured_text,)).start()

    def process_interview_flow(self, speech_text):
        # Format explicitly into typed elements to stay safe from Pydantic validations
        user_content = types.Content(role="user", parts=[types.Part.from_text(text=speech_text)])
        self.conversation_history.append(user_content)
        self.stream_from_gemini_primary(speech_text)

    # --- ROUTING LANE 1: GEMINI CORE ---
    def stream_from_gemini_primary(self, speech_text):
        try:
            system_content = types.Content(role="user", parts=[types.Part.from_text(text=INTERVIEW_SYSTEM_INSTRUCTION)])
            contents = [system_content] + self.conversation_history[-6:]
            
            response = self.ai_client.models.generate_content_stream(model='gemini-2.5-flash', contents=contents)
            
            full_response = ""
            for chunk in response:
                if chunk.text:
                    full_response += chunk.text
                    self.text_updated.emit(full_response)
            
            model_content = types.Content(role="model", parts=[types.Part.from_text(text=full_response)])
            self.conversation_history.append(model_content)
            
        except Exception as gemini_err:
            print(f"[LANE 1 OUTAGE]: Gemini hit a wall. Tripping Lane 2 (Groq)...")
            if self.groq_client:
                self.stream_from_groq_secondary(speech_text)
            else:
                self.stream_from_cohere_tertiary(speech_text)

    # --- ROUTING LANE 2: GROQ LLAMA-3 ---
    def stream_from_groq_secondary(self, prompt_text):
        try:
            messages = [
                {"role": "system", "content": INTERVIEW_SYSTEM_INSTRUCTION},
                {"role": "user", "content": prompt_text}
            ]
            stream = self.groq_client.chat.completions.create(
                model="llama3-70b-8192",  
                messages=messages,
                stream=True,
            )
            full_response = ""
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
                    self.text_updated.emit(full_response)
            
            model_content = types.Content(role="model", parts=[types.Part.from_text(text=full_response)])
            self.conversation_history.append(model_content)
            
        except Exception as groq_err:
            print(f"[LANE 2 OUTAGE]: Groq hit rate limits. Tripping Lane 3 (Cohere)...")
            self.stream_from_cohere_tertiary(prompt_text)

    # --- ROUTING LANE 3: COHERE COMMAND-R ---
    def stream_from_cohere_tertiary(self, prompt_text):
        try:
            response = self.cohere_client.chat_stream(
                model="command-r",
                message=prompt_text,
                preamble=INTERVIEW_SYSTEM_INSTRUCTION
            )
            full_response = ""
            for chunk in response:
                if chunk.event_type == "text-generation":
                    full_response += chunk.text
                    self.text_updated.emit(full_response)
                    
            model_content = types.Content(role="model", parts=[types.Part.from_text(text=full_response)])
            self.conversation_history.append(model_content)
            
        except Exception as cohere_err:
            print(f"[LANE 3 OUTAGE]: Cohere rate-limited. Activating Ultimate Safety Lane 4 (Mistral)...")
            self.stream_from_mistral_quaternary(prompt_text)

    # --- ROUTING LANE 4: MISTRAL LARGE ---
    def stream_from_mistral_quaternary(self, prompt_text):
        try:
            response = self.mistral_client.chat.stream(
                model="mistral-large-latest",
                messages=[
                    {"role": "system", "content": INTERVIEW_SYSTEM_INSTRUCTION},
                    {"role": "user", "content": prompt_text}
                ]
            )
            full_response = ""
            for chunk in response:
                if chunk.data.choices[0].delta.content:
                    full_response += chunk.data.choices[0].delta.content
                    self.text_updated.emit(full_response)
                    
            model_content = types.Content(role="model", parts=[types.Part.from_text(text=full_response)])
            self.conversation_history.append(model_content)
            
        except Exception as mistral_err:
            print(f"[CRITICAL TOTAL SYSTEM OUTAGE]: {mistral_err}")
            self.text_updated.emit("// Network Outage: All 4 active API nodes reported capacity rate limits.")

    # --- SYSTEM VISION HANDLING LOOP ---
    def get_gemini_vision_answer(self, image_bytes):
        try:
            image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/png")
            instruction_part = types.Part.from_text(text=INTERVIEW_SYSTEM_INSTRUCTION)
            vision_content = types.Content(role="user", parts=[instruction_part, image_part])
            
            response = self.ai_client.models.generate_content_stream(model='gemini-2.5-flash', contents=[vision_content])
            
            full_response = ""
            for chunk in response:
                if chunk.text:
                    full_response += chunk.text
                    self.text_updated.emit(full_response)
        except Exception as e:
            print(f"Vision Processing Network Error: {e}")
            self.text_updated.emit("// Vision processing error. Primary Gemini endpoint currently occupied.")


class FixedTextBrowser(QTextBrowser):
    """Overrides scroll bounds so mouse wheels scroll cleanly over layer-masked window contexts"""
    def wheelEvent(self, event):
        scrollbar = self.verticalScrollBar()
        if event.angleDelta().y() > 0:
            scrollbar.setValue(scrollbar.value() - scrollbar.singleStep() * 3)
        else:
            scrollbar.setValue(scrollbar.value() + scrollbar.singleStep() * 3)
        event.accept()


# --- COCKPIT OVERLAY VIEW INTERFACE ---
class InvisibleOverlay(QMainWindow):
    def __init__(self):
        super().__init__()
        self._old_pos = None
        self.init_ui()

    def init_ui(self):
        self.setGeometry(100, 100, 800, 450)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool)

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        
        layout = QGridLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.text_display = FixedTextBrowser(self)
        self.text_display.setHtml(
            "<div style='color: #5c6370; font-family: monospace; font-size: 14px;'>"
            "// AUTOMATIC TECHNICAL INTERVIEW COPILOT INITIALIZED<br>"
            "// • Isolated Pipeline active on target device: VB-Cable Output<br>"
            "// • Multi-Lane Network Shielding Enabled (Gemini -> Groq -> Cohere -> Mistral)<br>"
            "// • Focus overlay app window and press [Ctrl + S] to dump screen canvas into context data matrix."
            "</div>"
        )
        self.text_display.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        
        # Premium VS Code Matte Charcoal UI Template
        self.text_display.setStyleSheet("""
            QTextBrowser {
                color: #abb2bf;                
                font-family: 'Consolas', 'Fira Code', 'Courier New', monospace; 
                font-size: 14px; 
                line-height: 1.5;
                background-color: #1e1e24;     
                border: none;
                border-radius: 6px;
                padding: 16px;
            }
            QScrollBar:vertical {
                border: none;
                background: #18181c;
                width: 14px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #3e4451;
                min-height: 30px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical:hover {
                background: #4b5263;         
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                border: none;
                background: none;
            }
        """)
        layout.addWidget(self.text_display, 0, 0)

        # Secure explicitly locked bottom-right resize grip handle layout
        self.sizegrip = QSizeGrip(self)
        self.sizegrip.setFixedSize(18, 18)
        self.sizegrip.setStyleSheet("""
            QSizeGrip { 
                background-color: #282c34;     
                border-bottom-right-radius: 6px;
            }
            QSizeGrip:hover {
                background-color: #3e4451;     
            }
        """)
        layout.addWidget(self.sizegrip, 0, 0, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom)

        self.setStyleSheet("background-color: #FF00FF;") 
        self.show()
        self.apply_chroma_key_cloak()

        # Fire continuous runtime background execution processing engine loops
        self.engine = CopilotEngine()
        self.engine.text_updated.connect(self.update_overlay_text)
        self.engine.start()

    def apply_chroma_key_cloak(self):
        try:
            hwnd = int(self.winId())
            user32 = ctypes.windll.user32
            current_style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            user32.SetWindowLongW(hwnd, GWL_EXSTYLE, current_style | WS_EX_LAYERED)
            user32.SetLayeredWindowAttributes(hwnd, 0x00FF00FF, 0, LWA_COLORKEY)
            user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE)
        except Exception as e:
            print(f"Error applying window masks: {e}")

    def update_overlay_text(self, text):
        # Convert plain line breaks into clean paragraph spacing structures natively
        formatted = text.replace("\n", "<br>").replace("```", "")
        self.text_display.setHtml(f"<div style='color: #abb2bf;'>{formatted}</div>")

    # --- DESKTOP VISION SHORTCUT LISTENER INTERCEPT ---
    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key.Key_S and (event.modifiers() & Qt.KeyboardModifier.ControlModifier):
            self.capture_desktop_context()
            event.accept()
        else:
            super().keyPressEvent(event)

    def capture_desktop_context(self):
        self.text_display.setHtml("<span style='color: #e5c07b;'>[Vision Matrix Triggered] Reading screen frame buffers...</span>")
        screenshot = ImageGrab.grab()
        img_byte_arr = io.BytesIO()
        screenshot.save(img_byte_arr, format='PNG')
        raw_bytes = img_byte_arr.getvalue()
        
        threading.Thread(target=self.engine.get_gemini_vision_answer, args=(raw_bytes,)).start()

    # --- FRAME DRAG MANAGEMENT WINDOW MOVE OVERRIDES ---
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._old_pos = event.globalPosition().toPoint()

    def mouseMoveEvent(self, event):
        if self._old_pos is not None:
            delta = QPoint(event.globalPosition().toPoint() - self._old_pos)
            self.move(self.x() + delta.x(), self.y() + delta.y())
            self._old_pos = event.globalPosition().toPoint()

    def mouseReleaseEvent(self, event):
        self._old_pos = None


if __name__ == "__main__":
    app = QApplication(sys.argv)
    overlay = InvisibleOverlay()
    sys.exit(app.exec())