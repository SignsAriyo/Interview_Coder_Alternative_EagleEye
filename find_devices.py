import sounddevice as sd

print("--- ALL AVAILABLE AUDIO DEVICES ---")
devices = sd.query_devices()
for idx, d in enumerate(devices):
    print(f"Index {idx}: {d['name']}")
    print(f"   Max Inputs: {d['max_input_channels']} | Max Outputs: {d['max_output_channels']}")
    # Print Host API name if available
    try:
        host_api_name = sd.query_hostapis(d['hostapi'])['name']
        print(f"   Host API: {host_api_name}")
    except:
        pass
    print("-" * 30)