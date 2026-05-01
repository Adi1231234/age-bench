"""Detect grayscale/B&W images. Output: filename and the avg saturation (0-255).
Threshold: if max saturation < 25 → B&W."""
import os, sys
from PIL import Image

def avg_saturation(path):
    try:
        img = Image.open(path).convert('HSV')
        # Get S channel mean
        s = img.split()[1]
        # Sample (downscale for speed)
        s = s.resize((64, 64))
        pixels = list(s.getdata())
        return sum(pixels) / len(pixels)
    except Exception as e:
        return -1

THRESHOLD = 15  # avg saturation below this = B&W
roots = ['kids', 'adults']
for root in roots:
    if not os.path.isdir(root):
        continue
    for fn in sorted(os.listdir(root)):
        if not fn.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        path = os.path.join(root, fn)
        sat = avg_saturation(path)
        marker = '** B&W **' if sat < THRESHOLD else ''
        print(f'{path}\t{sat:.1f}\t{marker}')
