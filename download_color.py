"""Download photos from Pexels and verify they're color (not B&W)."""
import os, sys, urllib.request
from PIL import Image

def avg_saturation(path):
    try:
        img = Image.open(path).convert('HSV')
        s = img.split()[1].resize((64, 64))
        pixels = list(s.getdata())
        return sum(pixels) / len(pixels)
    except Exception:
        return -1

def download_and_verify(pid, target):
    url = f'https://images.pexels.com/photos/{pid}/pexels-photo-{pid}.jpeg?auto=compress&cs=tinysrgb&w=512'
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as r:
            data = r.read()
        with open(target, 'wb') as f:
            f.write(data)
        sat = avg_saturation(target)
        if sat < 25:
            os.remove(target)
            return False, f'B&W ({sat:.1f})'
        return True, f'OK ({sat:.1f})'
    except Exception as e:
        if os.path.exists(target):
            os.remove(target)
        return False, str(e)[:50]

# args: folder prefix start_idx pid1 pid2 ...
folder = sys.argv[1]
prefix = sys.argv[2]
start = int(sys.argv[3])
pids = sys.argv[4:]

idx = start
for pid in pids:
    target = f'{folder}/{prefix}-{idx}.jpg'
    if os.path.exists(target):
        idx += 1
        continue
    ok, msg = download_and_verify(pid, target)
    print(f'pid={pid} -> {target}: {msg}')
    if ok:
        idx += 1
