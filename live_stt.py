import contextlib
import json
import math
import threading
import time
import wave

import librosa
import numpy as np
import pyaudio
import whisper

from AudioInput import AudioRecorder
from overlap_text import combine_overlapping_text


class LiveSTT:

    frames = []
    buffer = []

    predicted_text = ""

    def __init__(self, processing_time=2, capture_time=8, whisper_model="small.en"):

        self.processing_time = processing_time
        self.capture_time = capture_time

        if self.processing_time >= self.capture_time:
            raise Exception("Capture time must be larger than processing time.")

        self.recorder = AudioRecorder()

        fs = self.recorder.frequency
        chunk = self.recorder.chunk

        self.capture_size = int(fs / chunk * self.capture_time)

        self._ticker = time.time()

        self.model = whisper.load_model(whisper_model)

        self.predicted_text = ""

        self._processing_thread = None

    def capture(self):
        data = self.recorder.record()

        self.frames.append(data)

    def process(self):
        self.capture()

        if time.time() - self._ticker > self.processing_time:

            if self._processing_thread is not None:
                print("Existing audio is still processing, skipping...")
                print(f"Lost approximately {self.processing_time}s of data.")
                return

            self._ticker = time.time()

            print("Processing...")

            process_data = self.frames[-self.capture_size:]

            self._processing_thread = threading.Thread(target=lambda: self.process_thread(process_data))
            self._processing_thread.start()

    def run_calibration(self, wave_file="calib_1.wav"):
        with contextlib.closing(wave.open(wave_file, 'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)

        t_start = time.time()
        self.model.transcribe(wave_file)

        process_time = time.time() - t_start

        return process_time/duration

    def calculate_recommended_settings(self, live_time=2, safety_factor=2, apply=True, calib_file="calib_1.wav"):
        print("Running calibration")
        live_factor = stt.run_calibration(calib_file)

        print(f"The processing time is {live_factor * 1000}ms per second of audio.")

        processing_time = live_time
        capture_time = (1 / live_factor * processing_time) / max(safety_factor, 1)

        if processing_time >= capture_time:
            if safety_factor > 1:
                raise Exception("GPU or CPU is not fast enough to do live transcription. Try decreasing the safety factor.")
            raise Exception("GPU or CPU is not fast enough to do live transcription.")

        if apply:
            self.capture_time = math.floor(capture_time)
            self.processing_time = processing_time

            print(f"Capture time is set to {int(stt.capture_time)}s.")
            print(f"Process time is set to {stt.processing_time}s.")

        return processing_time, capture_time, live_factor

    def transcribe(self, process_data: list):

        model = self.model

        data = process_data[0]

        for frame in process_data[1:]:
            data += frame

        audio_data = np.frombuffer(data, dtype=np.int16).flatten().astype(np.float32) / 32768.0

        audio_data = librosa.resample(audio_data, orig_sr=44100, target_sr=16000)

        audio = whisper.pad_or_trim(audio_data)

        mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)

        options = whisper.DecodingOptions(language="en")
        result = whisper.decode(self.model, mel, options)

        self.buffer.append({"text": result.text, "weight": math.exp(result.avg_logprob)})


    def build_transcript(self):

        result = self.buffer[-1]["text"]

        if len(self.predicted_text) == 0:
            self.predicted_text = result
        else:
            self.predicted_text = combine_overlapping_text(self.predicted_text, result)

            if self.predicted_text is None:
                self.buffer.append({"text": "silence"})
                self.predicted_text += "[Silence]"
                self.predicted_text += result

                print("None output, possibly slight pause")

        with open("out.json", "w") as f:
            f.write(json.dumps(self.buffer, indent=4))

        return self.predicted_text

    def process_thread(self, process_data):
        try:
            self.transcribe(process_data)
            self.build_transcript()

        except Exception as e:
            print(e)

        self._processing_thread = None

    def clear_data(self):
        self.buffer.clear()
        self.frames.clear()

    def reset_ticker(self):
        self._ticker = time.time()