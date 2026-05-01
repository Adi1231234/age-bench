"""Scrape Pexels search results for photo IDs. No API key needed."""
import urllib.request, re, sys

query = sys.argv[1]
page = int(sys.argv[2]) if len(sys.argv) > 2 else 1
url = f'https://www.pexels.com/search/{urllib.parse.quote(query)}/?page={page}'

import urllib.parse
url = f'https://www.pexels.com/search/{urllib.parse.quote(query)}/?page={page}'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0)'})
with urllib.request.urlopen(req, timeout=15) as r:
    html = r.read().decode('utf-8', errors='ignore')

# Find photo IDs in URLs like /photo/SLUG-NAME-12345/ or images.pexels.com/photos/12345/
ids = set()
for m in re.finditer(r'/photos/(\d+)/', html):
    ids.add(m.group(1))
for m in re.finditer(r'photos\.com/(\d+)/', html):
    ids.add(m.group(1))

for pid in sorted(ids, key=int):
    print(pid)
