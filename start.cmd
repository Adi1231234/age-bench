@echo off
cd /d "%~dp0"
echo Starting local server at http://localhost:8765
echo Open Chrome (recommended for WebGPU) and go to http://localhost:8765/
start "" "http://localhost:8765/"
python -m http.server 8765
