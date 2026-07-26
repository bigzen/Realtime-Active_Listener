import time
import socket
import pyaudio
import numpy as np
import threading

#pa constants
chunk = 2  # Record in chunks of 2 samples
sample_format = pyaudio.paFloat32  # 16 bits per sample
channels = 1
fs = 16000  # Record at 44100 samples per second
count = 0

#socket constants
wsl_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_address = ('172.18.236.8', 5052)

def audio_callback(indata, frames, time, status):
    #print(np.frombuffer(indata, dtype=np.float32))
    wsl_socket.sendto(indata, server_address)
    global count
    count += 1
    return (indata, pyaudio.paContinue)
    

if __name__ == "__main__":
    p = pyaudio.PyAudio()  # Create an interface to PortAudio
    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=chunk,
                    output_device_index=3,
                    input_device_index=1,
                    input=True,
                    output=False,
                    stream_callback=audio_callback)



    stream.start_stream()
    start_time = time.time()
            
    try:
        input("Press Enter to stop...\n")
        while stream.is_active():
            if count % 100 == 0:
                print(f"Sent {count} chunks of audio data.")
    except KeyboardInterrupt:
        print("\nStopped by user.")
        end_time = time.time()
        print(f"Recording duration: {end_time - start_time} seconds")
    finally:
        stream.stop_stream()
        stream.close()
        # Terminate the PortAudio interface
        p.terminate()     
