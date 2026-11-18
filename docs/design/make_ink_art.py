"""Static "deep ink" artwork for the walkthrough PDF and the slide deck.

A still version of the app's fluid background: marbled cyan -> blue -> violet ->
magenta ink on the near-black base #04050c. Needs numpy and Pillow.

    python docs/design/make_ink_art.py docs/design

Writes ink_cover.png (vivid, darker in the middle-left for a title) and
ink_haze.png (dim glows, like the app pages' background).
"""
import os
import sys

import numpy as np
from PIL import Image

W, H = 1920, 1080
BASE = np.array([4, 5, 12], dtype=np.float32) / 255.0
# Ink band (hue 0.5 -> 0.92 in the app's fluid), slightly toned for print/projector.
INKS = np.array([
    [12, 232, 255],   # cyan
    [60, 120, 255],   # blue
    [140, 70, 255],   # violet
    [235, 50, 225],   # magenta
], dtype=np.float32) / 255.0


def warp(x, y, rng, octaves=4, amp=0.18):
    """Domain warp with a few random sinusoids: gives the swirled, marbled look."""
    for o in range(octaves):
        f = 2.2 * (1.9 ** o)
        a = amp / (1.7 ** o)
        p1, p2, p3, p4 = rng.uniform(0, 2 * np.pi, 4)
        x, y = (x + a * np.sin(f * y + p1) * np.cos(0.7 * f * x + p2),
                y + a * np.sin(f * x + p3) * np.cos(0.8 * f * y + p4))
    return x, y


def paint(seed, n_blobs, strength, streaks, vignette, radius=(0.10, 0.32)):
    rng = np.random.default_rng(seed)
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    x = xs / H                 # aspect-correct units
    y = ys / H
    wx, wy = warp(x, y, rng)
    acc = np.zeros((H, W, 3), dtype=np.float32)
    for i in range(n_blobs):
        cx, cy = rng.uniform(-0.1, W / H + 0.1), rng.uniform(-0.1, 1.1)
        r = rng.uniform(*radius)
        d2 = (wx - cx) ** 2 + (wy - cy) ** 2
        blob = np.exp(-d2 / (r * r))
        col = INKS[i % len(INKS)] * rng.uniform(0.7, 1.0)
        acc += blob[..., None] * col
    # Oily streaks: modulate the ink with fine bands along the warped field.
    band = 0.5 + 0.5 * np.sin(38.0 * (wx * 0.6 + wy) + 3.0 * np.sin(9.0 * wx))
    acc *= (1.0 - streaks) + streaks * band[..., None]
    ink = 1.0 - np.exp(-strength * acc)      # soft tone-map, keeps colour, no clipping
    img = BASE + (1.0 - BASE) * ink * 0.92
    # Darken towards a focal point so text placed there stays legible.
    fx, fy, fr, depth = vignette
    d = np.sqrt((x - fx) ** 2 + (y - fy) ** 2)
    scrim = np.clip(1.0 - depth * np.exp(-(d / fr) ** 2), 0.0, 1.0)
    img = BASE + (img - BASE) * scrim[..., None]
    return Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))


def main(out):
    os.makedirs(out, exist_ok=True)
    paint(5, 22, 2.2, 0.55, (0.55, 0.52, 0.55, 0.88), radius=(0.05, 0.17)).save(os.path.join(out, "ink_cover.png"))
    paint(11, 8, 0.55, 0.15, (0.89, 0.5, 1.4, 0.55)).save(os.path.join(out, "ink_haze.png"))
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__)))
