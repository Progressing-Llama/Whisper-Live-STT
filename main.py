import os
import time

from live_stt_2 import LiveSTT

print("Initializing...")

stt = LiveSTT()
#stt.calculate_recommended_settings(5)

print("Running")

def clear_screen():
    # For Windows
    if os.name == 'nt':
        _ = os.system('cls')
    # For macOS and Linux
    else:
        _ = os.system('clear')

stt.start()

while True:
    time.sleep(2)
    clear_screen()

    print("Predicted Text:")
    print(stt.confirmed_text)