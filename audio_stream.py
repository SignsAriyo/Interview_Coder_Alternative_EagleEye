import sounddevice as sd
import numpy as np
import queue
import sys
from transcriber import AudioTranscriber

# Thread-safe queue to pass audio data from the background stream to our processing loop
audio_queue = queue.Queue()

BUFFER_DURATION = 3.5  # Chunk size to process (in seconds)

def audio_callback(indata, frames, time, status):
    """This function runs in the background for every millisecond of audio recorded."""
    if status:
        print(status, file=sys.stderr)
    # Put a copy of the audio data into our processing queue
    audio_queue.put(indata.copy())

def main():
    # Initialize our Whisper engine
    transcriber = AudioTranscriber()
    
    # --- AUTOMATIC HARDWARE DETECTION ---
    devices = sd.query_devices()
    target_idx = None
    device_name = ""

    # Strategy A: Look for Stereo Mix
    for idx, d in enumerate(devices):
        if d['max_input_channels'] > 0 and "stereo mix" in d['name'].lower():
            target_idx = idx
            device_name = d['name']
            break

    # Strategy B: Look for Loopback channels
    if target_idx is None:
        for idx, d in enumerate(devices):
            if d['max_input_channels'] > 0 and "loopback" in d['name'].lower():
                target_idx = idx
                device_name = d['name']
                break

    # Strategy C: Standard input backup
    if target_idx is None:
        default_input = sd.query_devices(kind='input')
        target_idx = default_input['index']
        device_name = default_input['name']

    chosen_device = devices[target_idx]
    
    # Dynamically extract parameters matching exactly what Windows wants
    channels_to_use = min(2, chosen_device['max_input_channels'])
    sample_rate = int(chosen_device['default_samplerate'])
    block_size = int(sample_rate * BUFFER_DURATION)
    
    print(f"\n[SYSTEM] Successfully bound stream engine onto: {device_name}")
    print(f"[SYSTEM] Hardware Parameters configured -> Sample Rate: {sample_rate}Hz | Channels: {channels_to_use}")
    
    # Array to pile up audio chunks until we have enough to transcribe
    recording_buffer = np.zeros((0, channels_to_use))

    # Start the hardware listener stream using dynamically resolved settings
    with sd.InputStream(device=target_idx, 
                        channels=channels_to_use, 
                        callback=audio_callback, 
                        samplerate=sample_rate):
        
        print("\n>>> LIVE TRANSCRIBER ACTIVE <<<")
        print("Play a YouTube video with clear English speech to test.")
        print("Press Ctrl+C to close.")
        
        try:
            while True:
                try:
                    chunk = audio_queue.get(timeout=0.1)
                    recording_buffer = np.vstack((recording_buffer, chunk))
                except queue.Empty:
                    continue

                if len(recording_buffer) >= block_size:
                    full_block = recording_buffer[:block_size]
                    recording_buffer = recording_buffer[block_size:]
                    
                    # Convert to mono if it's stereo layout
                    if channels_to_use == 2:
                        mono_audio = np.mean(full_block, axis=1)
                    else:
                        mono_audio = full_block.flatten()
                    
                    # Check if there's actual speech volume (ignore dead silence)
                    if np.linalg.norm(mono_audio) > 0.5:
                        print("[Processing...] ", end="", flush=True)
                        text = transcriber.transcribe_buffer(mono_audio, sample_rate=sample_rate)
                        
                        if text:
                            print(f"\nCaptured Speech: \"{text}\"\n")
                        else:
                            print("(No clear speech detected)")

        except KeyboardInterrupt:
            print("\nShutting down streaming engine safely.")

if __name__ == "__main__":
    main()