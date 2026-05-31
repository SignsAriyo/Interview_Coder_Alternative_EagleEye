import io
import wave
import numpy as np
from faster_whisper import WhisperModel

class AudioTranscriber:
    def __init__(self):
        print("Loading Whisper Speech-to-Text Engine...")
        # 'tiny.en' is extremely small, light on RAM, and blazing fast for English text.
        # compute_type="float16" uses your GPU if available, or "int8" for CPU-only efficiency.
        self.model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
        print("Transcription Engine Ready!")

    def transcribe_buffer(self, audio_data, sample_rate=44100):
        """Converts raw float32 arrays from sounddevice into text strings."""
        try:
            # 1. Convert float32 array to 16-bit PCM WAV bytes in memory
            # Whisper requires standard audio formatting to parse segments correctly
            byte_io = io.BytesIO()
            with wave.open(byte_io, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2) # 16-bit
                wav_file.setframerate(sample_rate)
                
                # Flatten audio array and convert to int16 format
                pcm_data = (audio_data * 32767).astype(np.int16)
                wav_file.writeframes(pcm_data.tobytes())
            
            byte_io.seek(0)

            # 2. Feed the in-memory WAV file directly to Faster-Whisper
            segments, info = self.model.transcribe(byte_io, beam_size=1)
            
            # Combine the chunks into a single readable string
            text_chunks = [segment.text for segment in segments]
            full_text = " ".join(text_chunks).strip()
            
            return full_text
        except Exception as e:
            print(f"Error during transcription process: {e}")
            return ""

# Basic isolated unit test
if __name__ == "__main__":
    # If run directly, it just checks if the model initializes smoothly
    engine = AudioTranscriber()