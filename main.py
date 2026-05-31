import sys
import queue
import threading
import numpy as np
import sounddevice as sd
import ctypes
import os
from pathlib import Path

from google import genai
from PyQt6.QtCore import Qt, QPoint, QThread, pyqtSignal
from PyQt6.QtGui import QCursor
from PyQt6.QtWidgets import QApplication, QTextBrowser, QMainWindow, QVBoxLayout, QWidget, QSizeGrip
from dotenv import load_dotenv

# Import your custom transcription worker
from transcriber import AudioTranscriber

# --- SECURITY & ENVIRONMENT INITIALIZATION ---
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / '.env'
load_dotenv(dotenv_path=ENV_PATH)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print(f"\n[ERROR] Could not find GEMINI_API_KEY inside environment variables.")
    sys.exit(1)

print(f"[SUCCESS] Security check passed. Secret Key loaded from local environment storage.")


# --- SPEED CONFIGURATION ---
CHUNK_DURATION = 1.0   
SILENCE_THRESHOLD = 0.4 
SILENCE_CHUNKS_LIMIT = 2 

# Win32 Constants
WDA_EXCLUDEFROMCAPTURE = 0x00000011
GWL_EXSTYLE = -20
WS_EX_LAYERED = 0x80000
LWA_COLORKEY = 0x00000001


# --- BACKGROUND WORKER THREAD ---
class CopilotEngine(QThread):
    text_updated = pyqtSignal(str) 

    def __init__(self):
        super().__init__()
        self.audio_queue = queue.Queue()
        self.transcriber = AudioTranscriber()
        self.ai_client = genai.Client(api_key=GEMINI_API_KEY)
        self.running = True
        self.full_response_history = ""

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        self.audio_queue.put(indata.copy())

    def run(self):
        devices = sd.query_devices()
        target_idx = None
        for idx, d in enumerate(devices):
            if d['max_input_channels'] > 0 and "stereo mix" in d['name'].lower():
                target_idx = idx
                break
        if target_idx is None:
            for idx, d in enumerate(devices):
                if d['max_input_channels'] > 0 and "loopback" in d['name'].lower():
                    target_idx = idx
                    break
        if target_idx is None:
            target_idx = sd.query_devices(kind='input')['index']

        chosen_device = devices[target_idx]
        channels = min(2, chosen_device['max_input_channels'])
        sample_rate = int(chosen_device['default_samplerate'])
        chunk_samples = int(sample_rate * CHUNK_DURATION)

        speech_buffer = np.zeros((0, channels))
        consecutive_silence_count = 0
        has_spoken = False

        print(f"[SYSTEM] Engine active on Input Index {target_idx}")

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
                            print(f"\n[Captured Sentences]: {captured_text}")
                            threading.Thread(target=self.get_gemini_answer, args=(captured_text,)).start()

    def get_gemini_answer(self, speech_text):
        prompt = (
            f"You are acting as an expert technical advisor helping a Principal Software Engineer. "
            f"Analyze the following question/statement and provide a comprehensive, fully detailed answer. "
            f"Include structural architecture insights, complete and robust code implementations, edge cases, "
            f"and trade-offs where applicable. Avoid high-level summaries; deliver highly technical, complete depth. "
            f"Context Audio: \"{speech_text}\""
        )
        try:
            response = self.ai_client.models.generate_content_stream(model='gemini-2.5-flash', contents=prompt)
            self.full_response_history = ""
            for chunk in response:
                if chunk.text:
                    self.full_response_history += chunk.text
                    self.text_updated.emit(self.full_response_history)
        except Exception as e:
            print(f"Gemini API Error: {e}")


# --- FOREGROUND OVERLAY ---
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
        
        # We leave a tiny bit of margin at the bottom right so the grip widget layout aligns nicely
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.text_display = QTextBrowser(self)
        self.text_display.setPlainText("Copilot active. Drag window body to move, drag bottom-right corner grip to resize...")
        
        self.text_display.setStyleSheet("""
            QTextBrowser {
                color: #00FF00; 
                font-family: 'Consolas', 'Courier New', monospace; 
                font-size: 14px; 
                font-weight: bold;
                background-color: #121212;
                border: none;
                border-radius: 5px;
                padding: 12px;
            }
            QScrollBar:vertical {
                border: none;
                background: #1e1e1e;
                width: 10px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #00FF00;
                min-height: 20px;
                border-radius: 5px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                border: none;
                background: none;
            }
        """)
        layout.addWidget(self.text_display)

        # --- THE PHYSICAL RESIZE GRIP HANDLE ---
        # This sits explicitly inside our interface layout window loop, capturing clicks
        self.sizegrip = QSizeGrip(self)
        self.sizegrip.setStyleSheet("""
            QSizeGrip {
                background-color: #00FF00; /* Solid green small rectangle block */
                width: 16px;
                height: 16px;
            }
        """)
        # Align it strictly to the bottom right corner of our app frame block
        layout.addWidget(self.sizegrip, 0, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom)

        self.setStyleSheet("background-color: #FF00FF;") 
        self.show()
        self.apply_chroma_key_cloak()

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
        self.text_display.setPlainText(text)

    # --- DRAG WINDOW OVERRIDES ---
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