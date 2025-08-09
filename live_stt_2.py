import json
import math
import queue
import string
import threading
import time
from collections import deque
from threading import Thread

import librosa
import numpy as np
import scipy
import webrtcvad
import whisper
from scipy.signal import resample

from AudioInput import AudioRecorder
from whisper_correction import intelligent_concatenate


class TimeFrame:
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.duration = end - start

    def intersects(self, other_timeframe):
        """
        Checks if this TimeFrame intersects with another TimeFrame.

        Two timeframes intersect if:
        1. The start of one is within the other.
        2. The end of one is within the other.
        3. One completely encompasses the other.
        """
        # Case 1: This timeframe starts within the other
        if self.start >= other_timeframe.start and self.start < other_timeframe.end:
            return True
        # Case 2: This timeframe ends within the other
        if self.end > other_timeframe.start and self.end <= other_timeframe.end:
            return True
        # Case 3: The other timeframe starts within this one (and potentially ends after this one)
        if other_timeframe.start >= self.start and other_timeframe.start < self.end:
            return True
        # Case 4: The other timeframe ends within this one (and potentially starts before this one)
        if other_timeframe.end > self.start and other_timeframe.end <= self.end:
            return True
        # Case 5: One completely encompasses the other (already covered by above, but for clarity)
        if self.start <= other_timeframe.start and self.end >= other_timeframe.end:
            return True
        if other_timeframe.start <= self.start and other_timeframe.end >= self.end:
            return True


        # Simplified condition for intersection:
        # If one timeframe starts after the other ends, they don't intersect.
        # Otherwise, they do.
        # return not (self.end <= other_timeframe.start or other_timeframe.end <= self.start)

        return False


class LiveSTT:

    def __init__(self, processing_time=4, capture_time=8, whisper_model="turbo"):
        self.processing_time = processing_time
        self.capture_time = capture_time


        self.recorder = AudioRecorder(chunk=2048)
        self.window_size = int(self.recorder.frequency / self.recorder.chunk * self.capture_time)
        self.step_size = int(self.recorder.frequency / self.recorder.chunk * self.processing_time)

        print(f"Window size: {self.window_size}")
        print(f"Step size: {self.step_size}")

        self.dequeued_audio = deque()
        self.transcribed_audio = queue.Queue()

        self.lock = threading.Lock()
        self.lock_transcript = threading.Lock()

        self.running = False

        self.recorder_thread = Thread(target=self.capture, daemon=True)
        self.processing_thread = Thread(target=self.process, daemon=True)
        self.builder_thread = Thread(target=self.build_transcript, daemon=True)

        self.model = whisper.load_model(whisper_model).to("cuda:0")

        self.predicted_text = ""
        self.confirmed_text = ""
        self.last_predicted = None
        self.last_predicted_text = ""

        self.current_predicted_text = ""

        self.start_time = time.time()

        self.vad_model = webrtcvad.Vad(2)

    def capture(self):
        while self.running:
            data = self.recorder.record()

            with self.lock:
                self.dequeued_audio.append((time.time() - self.start_time, data))

    def process(self):

        while self.running:

            window = []
            times = []

            with self.lock:
                if len(self.dequeued_audio) >= self.window_size:

                    for i in range(self.window_size):
                        window.append(self.dequeued_audio[i][1])
                        times.append(self.dequeued_audio[i][0])

                    for _ in range(self.step_size):
                        if self.dequeued_audio:
                            self.dequeued_audio.popleft()

            if len(window) > 0:

                speech = self.detect_speech_in_frames(window)
                if not speech:
                    continue

                time_frame = TimeFrame(min(times), max(times))
                self.transcribe(window, time_frame)

    def build_transcript(self):

        while self.running:
            time.sleep(0.1)
            try:
                with self.lock_transcript:
                    result = self.transcribed_audio.get(timeout=0.1)
            except queue.Empty:
                continue

            predicted_text = result["text"]
            weight = result["weight"]
            time_frame = result["time_frame"]

            if len(predicted_text) + len(self.confirmed_text) > 0:
                self.confirmed_text = intelligent_concatenate(self.confirmed_text, predicted_text)
            else:
                self.confirmed_text += predicted_text

            self.last_predicted = result

    def transcribe(self, process_data: list, time_frame: TimeFrame):

        model = self.model

        data = process_data[0]

        for frame in process_data[1:]:
            data += frame

        audio_data = np.frombuffer(data, dtype=np.int16).flatten().astype(np.float32) / 32768.0

        audio_data = librosa.resample(audio_data, orig_sr=44100, target_sr=16000)

        audio = whisper.pad_or_trim(audio_data)

        mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)

        if self.last_predicted is None:
            self.last_predicted = {"text": ""}

        options = whisper.DecodingOptions(language="en", suppress_tokens="", prefix=self.last_predicted["text"])
        result = whisper.decode(self.model, mel, options)
        print(result.text)

        out = {"text": result.text, "weight": math.exp(result.avg_logprob), "time_frame": time_frame}

        with self.lock_transcript:
            self.transcribed_audio.put(out)

        self.current_predicted_text = result.text

        return result.text

    def detect_speech_in_frames(self, audio_frames):
        """
        Detects speech in 44.1kHz audio frames using WebRTC VAD.

        Args:
            audio_frames (np.ndarray): Mono audio signal (float32 or int16), 44.1 kHz.

        Returns:
            List[bool]: True for speech, False for silence, one per frame.
        """

        data = audio_frames[0]

        for frame in audio_frames[1:]:
            data += frame

        audio_data = np.frombuffer(data, dtype=np.int16).flatten().astype(np.float32) / 32768.0

        audio_data = librosa.resample(audio_data, orig_sr=44100, target_sr=16000)

        audio_data *= 32767

        # Prepare frame chunks for VAD
        frame_duration_ms = 30  # 10, 20, or 30 ms supported
        frame_size = int(16000 * frame_duration_ms / 1000)
        byte_data = audio_data.tobytes()

        speech_flags = []
        for i in range(0, len(byte_data), frame_size * 2):  # 2 bytes per sample
            frame = byte_data[i:i + frame_size * 2]
            if len(frame) < frame_size * 2:
                break
            is_speech = self.vad_model.is_speech(frame, 16000)
            speech_flags.append(1 if is_speech else 0)

        return sum(speech_flags)/len(speech_flags) > 0.3

    def start(self):
        self.running = True
        self.start_time = time.time()

        self.recorder_thread.start()
        self.processing_thread.start()
        self.builder_thread.start()

    def stop(self):
        self.running = False

    def __del__(self):
        self.stop()
        self.recorder.close()