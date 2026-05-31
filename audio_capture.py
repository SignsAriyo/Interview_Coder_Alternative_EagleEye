import sounddevice as sd
import numpy as np

def callback(indata, frames, time, status):
    if status:
        print(status, flush=True)
    # Track the active channel audio volume level
    volume_norm = np.linalg.norm(indata) * 10
    if volume_norm > 0.05:  
        print(f"Captured Audio Level: {'|' * int(min(volume_norm * 5, 50))}")

try:
    devices = sd.query_devices()
    target_idx = None
    device_name = ""

    print("--- Scanning for Windows Recording Capture Channels ---")
    
    # Strategy A: Look for "Stereo Mix" (The most stable Windows recording route)
    for idx, d in enumerate(devices):
        if d['max_input_channels'] > 0 and "stereo mix" in d['name'].lower():
            target_idx = idx
            device_name = d['name']
            print(f"[FOUND] Utilizing Hardware Stereo Mix at Index {idx}")
            break

    # Strategy B: If Stereo Mix isn't enabled, look for the designated WASAPI loopback input device
    if target_idx is None:
        for idx, d in enumerate(devices):
            if d['max_input_channels'] > 0 and "loopback" in d['name'].lower():
                target_idx = idx
                device_name = d['name']
                print(f"[FOUND] Utilizing WASAPI Loopback Capture Interface at Index {idx}")
                break

    # Strategy C: Total Fallback (Use the standard OS default input mic system)
    if target_idx is None:
        default_input = sd.query_devices(kind='input')
        target_idx = default_input['index']
        device_name = default_input['name']
        print(f"[FALLBACK] No loopback found. Binding to Default Input Mic: {device_name}")

    chosen_device = devices[target_idx]
    channels_to_use = min(2, chosen_device['max_input_channels'])
    
    print(f"\nFinalized Connection Binding -> Device: {device_name} (Index: {target_idx})")
    print(f"Allocating System Channels: {channels_to_use}")

    with sd.InputStream(device=target_idx,
                        channels=channels_to_use,
                        callback=callback,
                        samplerate=int(chosen_device['default_samplerate'])):
        
        print("\nAudio capture hook online.")
        print("Play audio on your computer or speak into your microphone to test.")
        print("Press Ctrl+C to terminate.")
        while True:
            sd.sleep(1000)

except KeyboardInterrupt:
    print("\nAudio processing loop closed.")
except Exception as e:
    print(f"\nCritical Initialization Failure: {e}")