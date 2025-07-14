import pyaudio


class AudioRecorder:

    def __init__(self, frequency=44100, chunk=1024):
        self.p = pyaudio.PyAudio()
        p = self.p

        self.frequency = frequency

        self.chunk = chunk

        self.stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=self.frequency,
                        input=True, input_device_index=1)

        print(self.p.get_device_info_by_index(1))

    def record(self):
        stream = self.stream
        data = stream.read(self.chunk, exception_on_overflow=False)

        return data

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

    def silence_detection(self, frame):
        pass