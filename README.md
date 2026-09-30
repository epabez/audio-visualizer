# Audio Visualizer

A real-time, microphone-driven audio visualizer written in Python. It captures live audio with sounddevice, runs an FFT with NumPy, and renders a mirrored, glowing frequency spectrum with pygame-ce.

## Features

- Live mic input with low-latency rolling FFT
- Log-spaced frequency bands so bass and treble both get room
- Three modes: bars, dots, and lines, all mirrored horizontally and vertically
- Bloom-style glow effect

## Setup

Requires Python 3.10+. On Windows and macOS everything installs straight from pip. On Linux, install PortAudio first with `sudo apt install libportaudio2`.

```bash
pip install -r requirements.txt
python visualizer.py
```

If the wrong mic is picked up, run `python visualizer.py --list-devices` and pass `--device N`.

## Controls

| Key | Action |
| --- | --- |
| Space | Cycle mode |
| Up / Down | Adjust sensitivity |
| G | Toggle glow |
| Esc / Q | Quit |
