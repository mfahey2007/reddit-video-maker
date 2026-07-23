# Reddit Video Maker

Generates Reddit-story style short videos (TTS via local Kokoro, subtitles via Whisper word timing, screenshots rendered with PIL) and optionally uploads them to YouTube / Google Drive.

## Setup

```bash
brew install ffmpeg espeak-ng
pip3.10 install -r requirements.txt
```

## YouTube / Drive upload (optional)

1. Create a project in [Google Cloud Console](https://console.cloud.google.com/), enable the **YouTube Data API v3** and/or **Google Drive API**.
2. Create an OAuth **Desktop app** credential and download it as `credentials.json` into this folder.
3. Edit `YOUTUBE_CHANNEL_ID` and `GOOGLE_DRIVE_FOLDER_ID` near the top of `reddit_video_maker.py` with your own values.
4. On first run, a browser window will open to authorize access. This creates `token.json` / `youtube_token.json` locally — **do not commit these**, they grant ongoing upload access to your account. They're already excluded via `.gitignore`.

If you skip this setup, the script still renders videos locally — it just skips the upload step.

## Usage

```bash
python3.10 reddit_video_maker.py            # render the built-in demo story
python3.10 reddit_video_maker.py --test      # fast test render, no TTS
python3.10 reddit_video_maker.py --random    # pull a real story from Reddit
python3.10 reddit_video_maker.py --reddit AmItheAsshole
python3.10 reddit_video_maker.py --count 5 --random
```

Add your own background gameplay clips to `background_vids/` (not included — check licensing before using footage you don't own).
