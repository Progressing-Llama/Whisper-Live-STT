from live_stt import LiveSTT

print("Initializing...")

stt = LiveSTT()
#stt.calculate_recommended_settings(5)

print("Running")

stt.reset_ticker()

while True:
    stt.process()

    print(stt.confirmed_text)