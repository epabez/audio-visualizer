"""
Real-time audio visualizer.

Captures live microphone input with sounddevice, analyzes it with a NumPy FFT,
and renders a mirrored, glowing spectrum with pygame-ce.

Controls:
    SPACE      cycle mode (bars / dots / lines)
    UP / DOWN  adjust sensitivity
    G          toggle glow
    ESC / Q    quit
"""
import argparse
import sys
import threading

import numpy as np
import pygame
import sounddevice as sd

# Audio settings
RATE = 44100
CHUNK = 2048          # FFT window size (samples)
BLOCK = 512           # samples delivered per mic callback

# Visual settings
WIDTH, HEIGHT = 1280, 720
FPS = 60
NUM_BANDS = 64
MIN_FREQ, MAX_FREQ = 40, 16000
DB_FLOOR, DB_CEIL = -10.0, 45.0
DECAY = 0.88          # how fast levels fall (closer to 1 = slower)
BG = (6, 6, 14)
MODES = ["bars", "dots", "lines"]

WINDOW = np.hanning(CHUNK).astype(np.float32)


class MicBuffer:
    """Rolling buffer of the newest CHUNK samples, filled by the audio thread."""

    def __init__(self):
        self._data = np.zeros(CHUNK, dtype=np.float32)
        self._lock = threading.Lock()

    def callback(self, indata, frames, time_info, status):
        with self._lock:
            self._data = np.concatenate((self._data, indata[:, 0]))[-CHUNK:]

    def snapshot(self):
        with self._lock:
            return self._data.copy()


def list_devices():
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            print(f"[{i}] {dev['name']}")


def make_band_edges():
    """Log-spaced FFT bin boundaries so bass and treble get fair screen space."""
    freqs = np.fft.rfftfreq(CHUNK, 1 / RATE)
    edges = np.logspace(np.log10(MIN_FREQ), np.log10(MAX_FREQ), NUM_BANDS + 1)
    idx = np.searchsorted(freqs, edges)
    for i in range(1, len(idx)):  # every band gets at least one bin
        idx[i] = max(idx[i], idx[i - 1] + 1)
    return np.clip(idx, 0, len(freqs))


def spectrum_levels(samples, edges, gain_db):
    """FFT the buffer and return one 0..1 level per frequency band."""
    mags = np.abs(np.fft.rfft(samples * WINDOW))
    bands = np.array([mags[edges[i]:edges[i + 1]].mean() for i in range(NUM_BANDS)])
    db = 20 * np.log10(bands + 1e-9) + gain_db
    return np.clip((db - DB_FLOOR) / (DB_CEIL - DB_FLOOR), 0.0, 1.0)


def band_color(i, level, t):
    c = pygame.Color(0)
    hue = (200 + i / NUM_BANDS * 160 + t * 15) % 360
    c.hsva = (hue, 85, 35 + 65 * level, 100)
    return c


def draw(surface, levels, mode, t):
    """Draw the spectrum mirrored left/right and top/bottom around the center."""
    levels = [float(x) for x in levels]  # pygame-ce rejects NumPy scalars
    cx, cy = WIDTH / 2, HEIGHT / 2
    col_w = WIDTH / (2 * NUM_BANDS)
    max_h = HEIGHT * 0.45

    if mode == "lines":
        order = list(range(NUM_BANDS - 1, -1, -1)) + list(range(NUM_BANDS))
        xs = [cx - (i + 0.5) * col_w for i in range(NUM_BANDS - 1, -1, -1)]
        xs += [cx + (i + 0.5) * col_w for i in range(NUM_BANDS)]
        for k in range(len(order) - 1):
            a, b = order[k], order[k + 1]
            color = band_color(a, levels[a], t)
            for sign in (-1, 1):
                p1 = (xs[k], cy + sign * levels[a] * max_h)
                p2 = (xs[k + 1], cy + sign * levels[b] * max_h)
                pygame.draw.line(surface, color, p1, p2, 3)
        return

    for i, level in enumerate(levels):
        h = level * max_h
        color = band_color(i, level, t)
        for side in (-1, 1):
            x = cx + side * (i + 0.5) * col_w
            if mode == "bars":
                w = max(col_w - 3, 1)
                pygame.draw.rect(surface, color, (x - w / 2, cy - h, w, max(2 * h, 1)))
            else:  # dots
                r = 2 + level * col_w * 0.6
                pygame.draw.circle(surface, color, (x, cy - h), r)
                pygame.draw.circle(surface, color, (x, cy + h), r)


def add_glow(frame, target):
    """Cheap bloom: downscale, upscale (blur), and add back on top."""
    for factor in (4, 12):
        small = pygame.transform.smoothscale(frame, (WIDTH // factor, HEIGHT // factor))
        blur = pygame.transform.smoothscale(small, (WIDTH, HEIGHT))
        target.blit(blur, (0, 0), special_flags=pygame.BLEND_ADD)


def main():
    parser = argparse.ArgumentParser(description="Real-time mic audio visualizer")
    parser.add_argument("--device", type=int, help="input device index")
    parser.add_argument("--list-devices", action="store_true", help="list input devices and exit")
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    mic = MicBuffer()
    try:
        stream = sd.InputStream(samplerate=RATE, channels=1, dtype="float32",
                                blocksize=BLOCK, device=args.device,
                                callback=mic.callback)
        stream.start()
    except Exception as e:
        print(f"Could not open microphone: {e}\nTry --list-devices and --device N.")
        sys.exit(1)

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Audio Visualizer")
    frame = pygame.Surface((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 22)

    edges = make_band_edges()
    levels = np.zeros(NUM_BANDS)
    mode_idx, gain_db, glow = 0, 0.0, True

    running = True
    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key == pygame.K_SPACE:
                        mode_idx = (mode_idx + 1) % len(MODES)
                    elif event.key == pygame.K_UP:
                        gain_db += 3
                    elif event.key == pygame.K_DOWN:
                        gain_db -= 3
                    elif event.key == pygame.K_g:
                        glow = not glow

            target = spectrum_levels(mic.snapshot(), edges, gain_db)
            levels = np.maximum(target, levels * DECAY)  # instant attack, smooth fall

            t = pygame.time.get_ticks() / 1000
            frame.fill((0, 0, 0))
            draw(frame, levels, MODES[mode_idx], t)

            screen.fill(BG)
            screen.blit(frame, (0, 0), special_flags=pygame.BLEND_ADD)
            if glow:
                add_glow(frame, screen)

            hud = f"{MODES[mode_idx]}  |  gain {gain_db:+.0f} dB  |  glow {'on' if glow else 'off'}"
            screen.blit(font.render(hud, True, (120, 120, 150)), (12, 10))

            pygame.display.flip()
            clock.tick(FPS)
    finally:
        stream.stop()
        stream.close()
        pygame.quit()


if __name__ == "__main__":
    main()
