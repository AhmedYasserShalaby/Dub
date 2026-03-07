#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

echo "[1/4] Installing system tools with Homebrew (ffmpeg, whisper-cpp, yt-dlp)..."
brew list ffmpeg >/dev/null 2>&1 || brew install ffmpeg
brew list whisper-cpp >/dev/null 2>&1 || brew install whisper-cpp
brew list yt-dlp >/dev/null 2>&1 || brew install yt-dlp

echo "[2/4] Creating virtual environment..."
python3 -m venv .venv

echo "[3/4] Installing Python dependencies..."
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "[4/4] Setup complete."
echo "Activate with: source $PROJECT_DIR/.venv/bin/activate"
