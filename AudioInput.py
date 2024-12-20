import pyaudio


class AudioRecorder:

    def __init__(self):
        self.p = pyaudio.PyAudio()
        p = self.p

        self.frequency = 44100

        self.chunk = 1024

        self.stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=self.frequency,
                        input=True, input_device_index=1)

        print(self.p.get_device_info_by_index(1))

    def record(self):
        stream = self.stream
        data = stream.read(self.chunk)

        return data

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()