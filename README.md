# Whisper-Live-STT

OpenAI's whisper unfortunetly does not support live transcription. This repo is meant to fix that using a sliding window method.

Feel free to check the detailed article below on the explanations behind the sliding window method.

**How to make Whisper STT live transcription.**
[Part 1](https://medium.com/p/79c628984fc6)
[Part 2](https://medium.com/@pcb.it18/how-to-make-whisper-stt-live-transcription-part-2-5daa1dfa3be8)
## Installation

```bash
pip install -U openai-whisper
pip install pyaudio
pip install librosa
pip install numpy
```

Whisper requires ffmpeg, therefore the additional installation is required. 
```bash
# on Ubuntu or Debian
sudo apt update && sudo apt install ffmpeg

# on Arch Linux
sudo pacman -S ffmpeg

# on MacOS using Homebrew (https://brew.sh/)
brew install ffmpeg

# on Windows using Chocolatey (https://chocolatey.org/)
choco install ffmpeg

# on Windows using Scoop (https://scoop.sh/)
scoop install ffmpeg
```

Alternatively, you can copy the ffmpeg binary file to the working directory of your project.

## Usage/Examples

```python
from live_stt import LiveSTT

print("Initializing...")

stt = LiveSTT()

print("Running")

stt.reset_ticker()

while True:
    stt.process()
    print(stt.confirmed_text)
```
## FAQ

#### FileNotFoundError: [WinError 2] The system cannot find the file specified

This error occurs when ffmpeg is not found in the environmental path or found in working directory. Check installation guide.
