# Importing necessary libraries
import contextlib  # Provides utilities for working with context managers
import json  # For working with JSON data
import math  # For mathematical operations
import threading  # For running tasks in separate threads
import time  # For time-related functions
import wave  # For reading and writing WAV audio files

import librosa  # For audio processing and feature extraction
import numpy as np  # For numerical operations
import pyaudio  # For audio input/output
import whisper  # For speech-to-text transcription using OpenAI's Whisper model

from overlap_text import combine_overlapping_text  # Custom function to combine overlapping text segments


# Class for recording audio from a microphone
class AudioRecorder:

    def __init__(self):
        # Initialize PyAudio instance
        self.p = pyaudio.PyAudio()
        p = self.p

        # Set audio recording parameters
        self.frequency = 44100  # Sampling frequency
        self.chunk = 1024  # Number of audio frames per buffer

        # Open an audio stream for input
        self.stream = p.open(format=pyaudio.paInt16,  # 16-bit audio format
                             channels=1,  # Mono audio
                             rate=self.frequency,  # Sampling rate
                             input=True,  # Enable input
                             input_device_index=1)  # Use the second audio input device

        # Print information about the selected audio input device
        print(self.p.get_device_info_by_index(1))

    def record(self):
        # Record a chunk of audio data from the stream
        stream = self.stream
        data = stream.read(self.chunk)
        return data

    def close(self):
        # Stop and close the audio stream, and terminate PyAudio
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()


# Class for live speech-to-text transcription
class LiveSTT:

    # Initialize shared variables
    frames = []  # Stores audio frames
    buffer = []  # Stores transcription results
    predicted_text = ""  # Stores the final predicted text

    def __init__(self, processing_time=2, capture_time=8, whisper_model="small.en"):
        # Initialize parameters for processing and capturing audio
        self.processing_time = processing_time  # Time interval for processing audio
        self.capture_time = capture_time  # Time interval for capturing audio

        # Ensure capture time is greater than processing time
        if self.processing_time >= self.capture_time:
            raise Exception("Capture time must be larger than processing time.")

        # Initialize the audio recorder
        self.recorder = AudioRecorder()

        # Calculate the number of chunks to capture based on capture time
        fs = self.recorder.frequency
        chunk = self.recorder.chunk
        self.capture_size = int(fs / chunk * self.capture_time)

        # Initialize a ticker for timing
        self._ticker = time.time()

        # Load the Whisper model for transcription
        self.model = whisper.load_model(whisper_model)

        # Initialize a thread for processing audio
        self._processing_thread = None

    def capture(self):
        # Capture a chunk of audio data and store it in the frames list
        data = self.recorder.record()
        self.frames.append(data)

    def process(self):
        # Capture audio and process it if the processing interval has elapsed
        self.capture()

        if time.time() - self._ticker > self.processing_time:

            # Check if a processing thread is already running
            if self._processing_thread is not None:
                print("Existing audio is still processing, skipping...")
                print(f"Lost approximately {self.processing_time}s of data.")
                return

            # Reset the ticker
            self._ticker = time.time()

            print("Processing...")

            # Get the most recent audio frames for processing
            process_data = self.frames[-self.capture_size:]

            # Start a new thread to transcribe the audio
            self._processing_thread = threading.Thread(target=lambda: self.transcribe(process_data))
            self._processing_thread.start()

    def run_calibration(self, wave_file="calib_1.wav"):
        # Calibrate the transcription model using a WAV file
        with contextlib.closing(wave.open(wave_file, 'r')) as f:
            frames = f.getnframes()  # Total number of frames in the file
            rate = f.getframerate()  # Sampling rate of the file
            duration = frames / float(rate)  # Duration of the audio in seconds

        # Measure the time taken to transcribe the file
        t_start = time.time()
        self.model.transcribe(wave_file)
        process_time = time.time() - t_start

        # Return the processing time per second of audio
        return process_time / duration

    def calculate_recommended_settings(self, live_time=2, safety_factor=2, apply=True, calib_file="calib_1.wav"):
        # Calculate recommended settings for live transcription
        print("Running calibration")
        live_factor = stt.run_calibration(calib_file)

        print(f"The processing time is {live_factor * 1000}ms per second of audio.")

        # Calculate capture time based on calibration results
        processing_time = live_time
        capture_time = (1 / live_factor * processing_time) / max(safety_factor, 1)

        # Ensure capture time is greater than processing time
        if processing_time >= capture_time:
            if safety_factor > 1:
                raise Exception("GPU or CPU is not fast enough to do live transcription. Try decreasing the safety factor.")
            raise Exception("GPU or CPU is not fast enough to do live transcription.")

        # Apply the calculated settings if requested
        if apply:
            self.capture_time = math.floor(capture_time)
            self.processing_time = processing_time

            print(f"Capture time is set to {int(stt.capture_time)}s.")
            print(f"Process time is set to {stt.processing_time}s.")

        return processing_time, capture_time, live_factor

    def transcribe(self, process_data: list):
        # Transcribe the captured audio data
        try:
            model = self.model

            # Combine all audio frames into a single buffer
            data = process_data[0]
            for frame in process_data[1:]:
                data += frame

            # Convert audio data to a NumPy array and normalize it
            audio_data = np.frombuffer(data, dtype=np.int16).flatten().astype(np.float32) / 32768.0

            # Resample the audio to 16 kHz
            audio_data = librosa.resample(audio_data, orig_sr=44100, target_sr=16000)

            # Prepare the audio for Whisper model input
            audio = whisper.pad_or_trim(audio_data)
            mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)

            # Perform transcription using Whisper
            options = whisper.DecodingOptions(language="en")
            result = whisper.decode(self.model, mel, options)

            # Combine overlapping text or reset if necessary
            if len(self.buffer) == 0:
                self.predicted_text = result.text
            else:
                self.predicted_text = combine_overlapping_text(self.predicted_text, result.text)

                if self.predicted_text is None:
                    self.predicted_text = result.text
                    self.buffer.clear()
                    print("None output, resetting...")

            # Store the transcription result in the buffer
            self.buffer.append({"text": result.text, "weight": math.exp(result.avg_logprob)})

            # Save the buffer to a JSON file
            with open("out.json", "w") as f:
                f.write(json.dumps(self.buffer, indent=4))

            # Print the predicted text
            print(self.predicted_text)

        except Exception as e:
            # Handle any errors during transcription
            print(e)

        # Reset the processing thread
        self._processing_thread = None

    def clear_data(self):
        # Clear the buffer and frames
        self.buffer.clear()
        self.frames.clear()

    def reset_ticker(self):
        # Reset the ticker to the current time
        self._ticker = time.time()


# Main script
print("Initializing...")

# Create an instance of the LiveSTT class
stt = LiveSTT()
stt.calculate_recommended_settings(5)

print("Running")

# Reset the ticker and start processing audio
stt.reset_ticker()

while True:
    stt.process()
