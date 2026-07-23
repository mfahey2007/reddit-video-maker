#!/usr/bin/env python3.10
"""
Reddit-Style Video Maker with Auto Voice Selection + PIL Subtitles
TTS: Kokoro (local, free) -- no API keys needed.
Subtitles are burned via PIL into transparent PNG overlays -- no ffmpeg subtitle
filter involved, so no path-escaping issues on macOS.

Usage:
    python3.10 reddit_video_maker.py          # normal mode
    python3.10 reddit_video_maker.py --test   # test mode, no TTS used

Requirements:
    pip3.10 install openai-whisper pillow kokoro soundfile
    brew install ffmpeg espeak-ng
"""

import os
import sys
import subprocess
import re, random
import numpy as np
import soundfile as sf
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from concurrent.futures import ThreadPoolExecutor

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

# Add as many background videos as you like -- missing files are skipped at startup.
# Each segment picks one at random.
BG_DIR = Path(__file__).parent / "background_vids"

BACKGROUND_VIDEOS = [
    str(BG_DIR / "Minecraft.mp4"),
    str(BG_DIR / "GTA.mp4"),
    str(BG_DIR / "Satisfying.mp4"),
    str(BG_DIR / "Subway Surfers.mp4"),
]

# Kokoro voice names  (American English, lang_code='a')
VOICE_MALE   = "am_michael"   # American Male
VOICE_FEMALE = "af_heart"     # American Female

SUBTITLE_FONT_SIZE = 72
SUBTITLE_WORDS_PER_GROUP = 3

# Whisper model size for subtitle word timing.
# "tiny"  — ~3x faster than base, timing is slightly less precise (fine for subtitles)
# "base"  — more accurate, slower
WHISPER_MODEL_SIZE = "tiny"

# Video compression — controls output file size vs quality.
# For libx264:          lower CRF = bigger file/better quality (range 0–51, default 23)
# For h264_videotoolbox: maps CRF to an equivalent average bitrate
# Recommended values:
#   23 — high quality,  ~150–300 MB per minute
#   28 — good quality,  ~50–100 MB per minute  ← default
#   32 — smaller files, ~20–40 MB per minute  (fine for mobile/YouTube Shorts)
VIDEO_CRF = 28
VIDEO_AUDIO_BITRATE = "128k"

# ─────────────────────────────────────────────
# STORY
# ─────────────────────────────────────────────

STORY = {
    "subreddit": "r/relationship_advice",
    "title": "I (29F) just found out my husband (31M) of 3 years has an entirely different family in another state. A wife. Two kids. A mortgage. I found out at his funeral.",
    "author": "u/ThrowRA_StillInShock",
    "upvotes": "234.7k",
    "narrator_gender": "auto",
    "body": (
        "I need someone to tell me this is real because I feel like I am living inside a nightmare I cannot wake up from. "
        "My husband Daniel died in a car accident eight days ago. "
        "I was devastated. We had been together five years, married three. "
        "I thought he was the love of my life. "
        "At the funeral a woman walked in with two small children, a boy and a girl, maybe four and six years old. "
        "She walked straight up to the casket like she had every right to be there. "
        "I introduced myself as Daniel's wife. "
        "She looked at me and said she was Daniel's wife. "
        "We were both holding funeral programs with the same man's face on them. "
        "It came out fast after that. "
        "They had been married for seven years. The kids were his. "
        "She knew nothing about me. I knew nothing about her. "
        "He had an apartment with me in Chicago and a house with her in Nashville. "
        "He told her he traveled for work. He told me the same thing. "
        "I have been going through everything since then. "
        "The separate bank account I never knew about. The second phone plan. The calendar with two sets of holidays. "
        "He was meticulous. He never slipped once in five years. "
        "I don't know how to grieve someone I never actually knew. "
        "I don't know if I am a widow or just a woman who was lied to. "
        "I don't even know what to do with my own wedding ring."
    ),
    "comments": [
        {
            "author": "u/FamilyLawyerThrowaway",
            "upvotes": "187.3k",
            "text": "This is called bigamy and it is a crime. You are legally his wife if you were married first. Get a lawyer before you do anything else. There may be life insurance and assets you are entitled to that the other family is already moving on."
        },
        {
            "author": "u/ISurvivedSomethingLikeThis",
            "upvotes": "143.6k",
            "text": "I found out my ex had a second family after six years. The grief is unlike anything else because you are mourning the person you thought they were AND the life you thought you had at the same time. Please be so gentle with yourself right now."
        },
        {
            "author": "u/JustNeedToSayThis99",
            "upvotes": "98.2k",
            "text": "The other woman is also a victim here. Two families destroyed by one person's lies. I hope you are both able to find some peace. This is not something either of you deserved."
        },
    ]
}

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

WIDTH = 1080
HEIGHT = 1920
BG_COLOR = (26, 26, 27)
CARD_COLOR = (30, 30, 30)
TEXT_COLOR = (215, 218, 220)
SUBTEXT_COLOR = (129, 131, 132)
ORANGE = (255, 69, 0)
FONT_SIZE_TITLE = 38
FONT_SIZE_BODY = 32
FONT_SIZE_COMMENT = 30
FONT_SIZE_META = 26
PADDING = 48
LINE_SPACING = 8

# YouTube upload
# Privacy: "public", "private", or "unlisted"
# Note: YouTube Data API v3 has a free quota of 10,000 units/day.
# Each video upload costs 1,600 units — so ~6 uploads/day on the free tier.
# Enable "YouTube Data API v3" in the same Google Cloud project as Drive.
YOUTUBE_PRIVACY = "public"
YOUTUBE_CATEGORY_ID = "22"   # 22 = People & Blogs, 24 = Entertainment
# Set this to the channel ID of the channel you manage.
# Find it: go to the channel on YouTube → About → Share → Copy channel ID
# Looks like: UCxxxxxxxxxxxxxxxxxxxxxxxxx
YOUTUBE_CHANNEL_ID = "UCxxxxxxxxxxxxxxxxxxxxxxxxx"

# Google Drive upload
# Set this to your Drive folder ID (the long string in the folder's URL).
# Leave blank ("") to upload to the root of My Drive instead.
# Requires: pip3.10 install google-api-python-client google-auth-httplib2 google-auth-oauthlib
# And a credentials.json file from Google Cloud Console (Drive API enabled, OAuth Desktop app).
GOOGLE_DRIVE_FOLDER_ID = ""

OUTPUT_DIR = Path("reddit_video_output")
SCREENSHOTS_DIR = OUTPUT_DIR / "screenshots"
AUDIO_DIR = OUTPUT_DIR / "audio"
SUBTITLE_DIR = OUTPUT_DIR / "subtitles"


def setup_dirs():
    VIDS_DIR.mkdir(exist_ok=True)
    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    SUBTITLE_DIR.mkdir(parents=True, exist_ok=True)


VIDS_DIR = Path(__file__).parent / "Vids"


def get_next_video_number():
    VIDS_DIR.mkdir(exist_ok=True)
    existing = list(VIDS_DIR.glob("vid*.mp4"))
    numbers = []
    for f in existing:
        match = re.match(r"vid(\d+)\.mp4", f.name)
        if match:
            numbers.append(int(match.group(1)))
    next_num = max(numbers) + 1 if numbers else 1
    return VIDS_DIR / f"vid{next_num}.mp4"


def detect_gender(story):
    gender = story.get("narrator_gender", "auto").lower()
    if gender == "male":
        print(f"🎙️  Voice: Male ({VOICE_MALE})")
        return VOICE_MALE
    if gender == "female":
        print(f"🎙️  Voice: Female ({VOICE_FEMALE})")
        return VOICE_FEMALE

    title = story.get("title", "").lower()
    body  = story.get("body", "").lower()
    # Only check the first 300 chars of body — narrator usually identifies early
    text  = title + " " + body[:300]

    # ── Tier 1: explicit age+gender tag e.g. (29F) (31M) ─────────────
    match = re.search(r'\((\d+)([mf])\)', text)
    if match:
        letter = match.group(2)
        if letter == "m":
            print(f"🎙️  Auto-detected: Male  [age tag]")
            return VOICE_MALE
        else:
            print(f"🎙️  Auto-detected: Female  [age tag]")
            return VOICE_FEMALE

    # ── Tier 2: explicit self-identification ──────────────────────────
    male_self = [
        r"i'?m a (man|guy|male|husband|boyfriend|father|dad|brother|son)\b",
        r"i am a (man|guy|male|husband|boyfriend|father|dad|brother|son)\b",
        r"as a (man|guy|male)\b",
        r"\b(male|man|guy) here\b",
        r"i \(m\b",
    ]
    female_self = [
        r"i'?m a (woman|girl|female|wife|girlfriend|mother|mom|sister|daughter)\b",
        r"i am a (woman|girl|female|wife|girlfriend|mother|mom|sister|daughter)\b",
        r"as a (woman|girl|female)\b",
        r"\b(female|woman|girl) here\b",
        r"i \(f\b",
    ]
    for pattern in male_self:
        if re.search(pattern, text):
            print(f"🎙️  Auto-detected: Male  [self-identification]")
            return VOICE_MALE
    for pattern in female_self:
        if re.search(pattern, text):
            print(f"🎙️  Auto-detected: Female  [self-identification]")
            return VOICE_FEMALE

    # ── Tier 3: narrator relationship keywords ────────────────────────
    # "my wife/girlfriend" → narrator is male; "my husband/boyfriend" → female
    # Weighted scoring so multiple hits build confidence
    male_score = len(re.findall(
        r'\b(my wife|my ex-wife|my girlfriend|my ex-girlfriend|'
        r'my daughter|my mom|my mother|my sister)\b', text))
    female_score = len(re.findall(
        r'\b(my husband|my ex-husband|my boyfriend|my ex-boyfriend|'
        r'my son|my dad|my father|my brother)\b', text))

    if male_score > female_score:
        print(f"🎙️  Auto-detected: Male  [relationship keywords]")
        return VOICE_MALE
    if female_score > male_score:
        print(f"🎙️  Auto-detected: Female  [relationship keywords]")
        return VOICE_FEMALE

    # ── Fallback: default female (relationship subs skew heavily female) ─
    print(f"🎙️  No gender signal — defaulting to Female")
    return VOICE_FEMALE


def load_font(size):
    font_paths = [
        "/Library/Fonts/Impact.ttf",
        "/System/Library/Fonts/Impact.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for path in font_paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def wrap_text(text, font, max_width, draw):
    words = text.split()
    lines = []
    current = ""
    for word in words:
        test = (current + " " + word).strip()
        bbox = draw.textbbox((0, 0), test, font=font)
        if bbox[2] <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def draw_rounded_rect(draw, xy, radius, fill):
    x1, y1, x2, y2 = xy
    draw.rectangle([x1 + radius, y1, x2 - radius, y2], fill=fill)
    draw.rectangle([x1, y1 + radius, x2, y2 - radius], fill=fill)
    draw.ellipse([x1, y1, x1 + 2*radius, y1 + 2*radius], fill=fill)
    draw.ellipse([x2 - 2*radius, y1, x2, y1 + 2*radius], fill=fill)
    draw.ellipse([x1, y2 - 2*radius, x1 + 2*radius, y2], fill=fill)
    draw.ellipse([x2 - 2*radius, y2 - 2*radius, x2, y2], fill=fill)


def make_post_screenshot():
    font_title = load_font(FONT_SIZE_TITLE)
    font_body = load_font(FONT_SIZE_BODY)
    font_meta = load_font(FONT_SIZE_META)

    dummy_img = Image.new("RGB", (WIDTH, 100))
    dummy_draw = ImageDraw.Draw(dummy_img)
    inner_w = WIDTH - PADDING * 2

    title_lines = wrap_text(STORY["title"], font_title, inner_w, dummy_draw)
    body_lines = wrap_text(STORY["body"], font_body, inner_w, dummy_draw)

    title_h = len(title_lines) * (FONT_SIZE_TITLE + LINE_SPACING)
    body_h = len(body_lines) * (FONT_SIZE_BODY + LINE_SPACING)
    total_h = PADDING + 40 + PADDING//2 + title_h + PADDING//2 + body_h + PADDING + 60

    img = Image.new("RGB", (WIDTH, total_h), BG_COLOR)
    draw = ImageDraw.Draw(img)
    draw_rounded_rect(draw, (PADDING//2, PADDING//2, WIDTH - PADDING//2, total_h - PADDING//2), 16, CARD_COLOR)

    y = PADDING
    draw.text((PADDING, y), f"{STORY['subreddit']}  •  Posted by {STORY['author']}", font=font_meta, fill=SUBTEXT_COLOR)
    y += 40 + PADDING // 2

    for line in title_lines:
        draw.text((PADDING, y), line, font=font_title, fill=TEXT_COLOR)
        y += FONT_SIZE_TITLE + LINE_SPACING
    y += PADDING // 2

    for line in body_lines:
        draw.text((PADDING, y), line, font=font_body, fill=TEXT_COLOR)
        y += FONT_SIZE_BODY + LINE_SPACING
    y += PADDING

    draw.text((PADDING, y), f"▲  {STORY['upvotes']}  •  Share  •  Save", font=font_meta, fill=ORANGE)

    path = SCREENSHOTS_DIR / "post.png"
    img.save(path)
    print(f"✅ Saved post screenshot: {path}")
    return path


def make_comment_screenshot(comment, index):
    font_text = load_font(FONT_SIZE_COMMENT)
    font_meta = load_font(FONT_SIZE_META)

    dummy_img = Image.new("RGB", (WIDTH, 100))
    dummy_draw = ImageDraw.Draw(dummy_img)
    inner_w = WIDTH - PADDING * 3

    lines = wrap_text(comment["text"], font_text, inner_w, dummy_draw)
    text_h = len(lines) * (FONT_SIZE_COMMENT + LINE_SPACING)
    total_h = PADDING + 36 + PADDING // 2 + text_h + PADDING

    img = Image.new("RGB", (WIDTH, total_h), BG_COLOR)
    draw = ImageDraw.Draw(img)
    draw_rounded_rect(draw, (PADDING//2, PADDING//2, WIDTH - PADDING//2, total_h - PADDING//2), 16, CARD_COLOR)
    draw.rectangle([PADDING, PADDING, PADDING + 4, total_h - PADDING], fill=ORANGE)

    y = PADDING
    x = PADDING + 20
    draw.text((x, y), f"{comment['author']}  •  {comment['upvotes']} points", font=font_meta, fill=SUBTEXT_COLOR)
    y += 36 + PADDING // 2

    for line in lines:
        draw.text((x, y), line, font=font_text, fill=TEXT_COLOR)
        y += FONT_SIZE_COMMENT + LINE_SPACING

    path = SCREENSHOTS_DIR / f"comment_{index}.png"
    img.save(path)
    print(f"✅ Saved comment screenshot: {path}")
    return path


def generate_tts(text, filename, voice, pipeline):
    """Generate TTS using Kokoro (local). Saves as 24 kHz WAV."""
    chunks = []
    for result in pipeline(text, voice=voice, speed=1.0):
        chunks.append(result.audio)
    if not chunks:
        print("❌ Kokoro returned no audio -- check voice name or text.")
        sys.exit(1)
    audio = np.concatenate(chunks)
    path = AUDIO_DIR / filename
    sf.write(str(path), audio, 24000)
    print(f"✅ Saved audio: {path}  ({len(audio)/24000:.1f}s)")
    return path


def generate_dummy_audio(duration_secs, filename):
    """Generate a silent WAV file for testing -- no TTS used."""
    path = AUDIO_DIR / filename
    subprocess.run([
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
        "-t", str(duration_secs),
        str(path)
    ], check=True, capture_output=True)
    print(f"✅ Saved dummy audio: {path}")
    return path


def make_dummy_timings(text, duration_secs):
    """Generate evenly spaced fake word timings for test mode."""
    words = text.split()
    if not words:
        return []
    step = duration_secs / len(words)
    return [
        {"word": w, "start": i * step, "end": (i + 1) * step}
        for i, w in enumerate(words)
    ]


def transcribe_with_whisper(audio_path, model):
    print(f"   🔍 Transcribing {audio_path.name}...")
    try:
        from faster_whisper import WhisperModel
        if isinstance(model, WhisperModel):
            segments, _ = model.transcribe(str(audio_path), word_timestamps=True)
            word_timings = []
            for seg in segments:
                for w in seg.words:
                    word_timings.append({"word": w.word.strip(), "start": w.start, "end": w.end})
            return word_timings
    except ImportError:
        pass
    # Fallback: openai-whisper
    result = model.transcribe(str(audio_path), word_timestamps=True)
    word_timings = []
    for segment in result["segments"]:
        for word_info in segment.get("words", []):
            word_timings.append({
                "word": word_info["word"].strip(),
                "start": word_info["start"],
                "end": word_info["end"]
            })
    return word_timings


SUBTITLE_STRIP_H  = SUBTITLE_FONT_SIZE + 24   # thin strip -- just tall enough for text
SUBTITLE_Y_BOTTOM = int(HEIGHT * 0.72)        # position while post card is visible
SUBTITLE_Y_MID    = int(HEIGHT * 0.47)        # position after post card disappears


def create_subtitle_frames(word_timings, segment_name):
    """
    Render each word group as a narrow RGBA PNG strip (WIDTH × SUBTITLE_STRIP_H).
    Using PIL stroke_width instead of a manual outline loop, and tiny strips instead
    of full 1080×1920 frames -- both dramatically cut runtime.
    Returns list of (path, start_sec, end_sec).
    """
    font   = load_font(SUBTITLE_FONT_SIZE)
    stroke = max(1, SUBTITLE_FONT_SIZE // 18)   # ~4 px at size 72
    frames = []
    seen   = {}   # text → path (dedup identical subtitle images)
    i = 0
    group_idx = 0

    while i < len(word_timings):
        group = word_timings[i:i + SUBTITLE_WORDS_PER_GROUP]
        start = group[0]["start"]
        end   = group[-1]["end"]
        text  = " ".join(w["word"] for w in group).upper()

        if text in seen:
            # Reuse the already-rendered PNG
            frames.append((seen[text], start, end))
        else:
            # Narrow strip -- full width, just tall enough for text + stroke
            img  = Image.new("RGBA", (WIDTH, SUBTITLE_STRIP_H), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)

            bbox   = draw.textbbox((0, 0), text, font=font,
                                   stroke_width=stroke)
            text_w = bbox[2] - bbox[0]
            x = (WIDTH - text_w) // 2
            y = stroke  # small top padding equal to stroke size

            # Single draw.text call with native stroke -- replaces the 80-call loop
            draw.text(
                (x, y), text, font=font,
                fill=(255, 255, 255, 255),
                stroke_width=stroke,
                stroke_fill=(0, 0, 0, 255),
            )

            path = SUBTITLE_DIR / f"{segment_name}_sub_{group_idx}.png"
            img.save(path)
            seen[text] = path
            frames.append((path, start, end))

        i += SUBTITLE_WORDS_PER_GROUP
        group_idx += 1

    unique = len(seen)
    print(f"   ✅ {len(frames)} subtitle frame(s) for '{segment_name}' ({unique} unique PNG(s))")
    return frames


def _render_subtitle_video(subtitle_frames, duration, segment_name):
    """
    Compile all subtitle strip PNGs into a single transparent video by piping
    raw RGBA frames to ffmpeg at 10 fps.

    Why: having 100+ individual PNG inputs each needs its own swscaler thread,
    exhausting OS resources (exit 232 / EAGAIN).  One video = one swscaler.

    Encoding: 'qtrle' (QuickTime Animation) is lossless, natively supports
    RGBA on macOS, and produces a pixel-for-pixel transparent overlay track.
    """
    if not subtitle_frames:
        return None

    fps          = 10
    total_frames = int(duration * fps) + 1
    out_path     = SUBTITLE_DIR / f"{segment_name}_subs.mov"

    # Pre-load strip images into raw bytes (they're tiny — WIDTH × SUBTITLE_STRIP_H)
    cache = {}
    for sub_path, _, _ in subtitle_frames:
        key = str(sub_path)
        if key not in cache:
            img = Image.open(key).convert("RGBA")
            cache[key] = img.tobytes()

    blank = bytes(WIDTH * SUBTITLE_STRIP_H * 4)   # fully transparent frame

    proc = subprocess.Popen(
        [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgba",
            "-s", f"{WIDTH}x{SUBTITLE_STRIP_H}",
            "-r", str(fps),
            "-i", "pipe:0",
            "-c:v", "qtrle",        # lossless RGBA, native macOS
            "-t", str(duration),
            str(out_path),
        ],
        stdin=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )

    for fi in range(total_frames):
        t     = fi / fps
        frame = blank
        for sub_path, start, end in subtitle_frames:
            if start <= t < end:
                frame = cache[str(sub_path)]
                break
        proc.stdin.write(frame)

    proc.stdin.close()
    proc.wait()
    print(f"   ✅ Subtitle video rendered: {out_path.name}")
    return out_path


def get_video_encoder():
    """Use Apple VideoToolbox GPU encoder on macOS, fall back to libx264."""
    if sys.platform == "darwin":
        r = subprocess.run(["ffmpeg", "-encoders"], capture_output=True, text=True)
        if "h264_videotoolbox" in r.stdout:
            return "h264_videotoolbox"
    return "libx264"


VIDEO_ENCODER = get_video_encoder()


def get_audio_duration(path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True
    )
    return float(result.stdout.strip())


def _render_screenshot_fade_video(screenshot_path, new_w, new_h, duration,
                                   fade_start, fade_end, segment_name):
    """
    Pre-render the post screenshot as a qtrle RGBA video with a PIL/numpy
    alpha fade baked in.  Much more reliable than filter-complex geq expressions.
    The card stays opaque until fade_start, then fades to invisible by fade_end.
    """
    fps          = 10
    total_frames = int(duration * fps) + 1
    out_path     = SUBTITLE_DIR / f"{segment_name}_post.mov"

    ss_arr = np.array(
        Image.open(screenshot_path).convert("RGBA").resize((new_w, new_h), Image.LANCZOS),
        dtype=np.uint8,
    )
    blank = bytes(new_w * new_h * 4)

    proc = subprocess.Popen(
        [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgba",
            "-s", f"{new_w}x{new_h}",
            "-r", str(fps),
            "-i", "pipe:0",
            "-c:v", "qtrle",
            "-t", str(duration),
            str(out_path),
        ],
        stdin=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )

    for fi in range(total_frames):
        t = fi / fps
        if t >= fade_end:
            proc.stdin.write(blank)
        elif t <= fade_start:
            proc.stdin.write(ss_arr.tobytes())
        else:
            alpha_mult = (fade_end - t) / (fade_end - fade_start)
            frame = ss_arr.copy()
            frame[:, :, 3] = (frame[:, :, 3] * alpha_mult).astype(np.uint8)
            proc.stdin.write(frame.tobytes())

    proc.stdin.close()
    proc.wait()
    print(f"   ✅ Post fade video: {out_path.name}")
    return out_path


def make_segment_video(screenshot_path, audio_path, subtitle_frames, output_path,
                       bg_video):
    """
    Compose one segment video.
    All alpha effects (screenshot fade, subtitle transparency) are pre-rendered
    in Python so ffmpeg only does simple overlay compositing — no geq/fade
    filter expressions that break across platforms.

    Inputs:  0=bg  1=post_fade  2=audio  [3=subtitles]
    """
    # Hard-coded fade: card visible for 3 s, fades out over 0.5 s
    _SHOW  = 3.0
    _FADE  = 0.5

    duration    = get_audio_duration(audio_path)
    bg_duration = get_audio_duration(bg_video)
    seek = random.randint(0, max(0, int(bg_duration) - int(duration) - 10))
    print(f"   🎥 Background: {Path(bg_video).name}  (seek {seek}s)")

    img_obj  = Image.open(screenshot_path)
    scale    = (WIDTH - 80) / img_obj.width
    new_w    = int(img_obj.width * scale)
    new_h    = int(img_obj.height * scale)
    y_offset = max(60, (HEIGHT - new_h) // 2 - 200)

    fade_start = min(_SHOW, max(0.0, duration - _FADE))
    fade_end   = fade_start + _FADE

    # Pre-render post fade and subtitle video in parallel — both are independent
    with ThreadPoolExecutor(max_workers=2) as ex:
        f_post = ex.submit(
            _render_screenshot_fade_video,
            screenshot_path, new_w, new_h, duration, fade_start, fade_end,
            output_path.stem,
        )
        f_sub = ex.submit(_render_subtitle_video, subtitle_frames, duration, output_path.stem)
        post_vid  = f_post.result()
        sub_video = f_sub.result()

    # Subtitle strip slides up from BOTTOM → MID while the card fades
    yb, ym = SUBTITLE_Y_BOTTOM, SUBTITLE_Y_MID
    y_expr = (
        f"if(lt(t,{fade_start:.3f}),"
        f"{yb},"
        f"if(lt(t,{fade_end:.3f}),"
        f"{yb}+({ym}-{yb})*(t-{fade_start:.3f})/{_FADE:.3f},"
        f"{ym}))"
    )

    if sub_video:
        filter_complex = (
            f"[0:v]scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase,"
            f"crop={WIDTH}:{HEIGHT},setpts=PTS-STARTPTS[bg];"
            f"[bg][1:v]overlay=40:{y_offset}:format=auto,format=yuv420p[base];"
            f"[base][3:v]overlay=x=0:y='{y_expr}':eval=frame:format=auto[vout]"
        )
    else:
        filter_complex = (
            f"[0:v]scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase,"
            f"crop={WIDTH}:{HEIGHT},setpts=PTS-STARTPTS[bg];"
            f"[bg][1:v]overlay=40:{y_offset}:format=auto,format=yuv420p[vout]"
        )

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(seek), "-stream_loop", "-1",
        "-i", str(Path(bg_video).resolve()),   # 0: bg
        "-i", str(post_vid.resolve()),          # 1: post fade
        "-i", str(audio_path.resolve()),        # 2: audio
    ]
    if sub_video:
        cmd += ["-i", str(sub_video.resolve())]  # 3: subtitles

    # Map CRF to an approximate bitrate for VideoToolbox (doesn't support -crf)
    _CRF_TO_BITRATE = {range(0, 24): "6M", range(24, 28): "4M",
                       range(28, 31): "3M", range(31, 35): "2M",
                       range(35, 52): "1M"}
    vt_bitrate = next((v for k, v in _CRF_TO_BITRATE.items() if VIDEO_CRF in k), "3M")

    encoder_flags = (
        ["-b:v", vt_bitrate]
        if VIDEO_ENCODER == "h264_videotoolbox"
        else ["-crf", str(VIDEO_CRF), "-preset", "fast"]
    )

    cmd += [
        "-filter_complex", filter_complex,
        "-map", "[vout]",
        "-map", "2:a",
        "-t", str(duration),
        "-c:v", VIDEO_ENCODER,
        *encoder_flags,
        "-c:a", "aac",
        "-b:a", VIDEO_AUDIO_BITRATE,
        str(output_path.resolve()),
    ]

    subprocess.run(cmd, check=True)
    print(f"✅ Segment video: {output_path}")
    return output_path


def _fmt_score(n):
    """Format an integer score as '47.3k' etc."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1000:
        return f"{n / 1000:.1f}k"
    return str(n)


def _clean_markdown(text):
    """Strip Reddit markdown so TTS reads naturally."""
    # links → just the display text
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    # bold / italic
    text = re.sub(r'\*{1,3}([^*\n]+)\*{1,3}', r'\1', text)
    # headers
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    # blockquotes
    text = re.sub(r'^>\s?', '', text, flags=re.MULTILINE)
    # horizontal rules
    text = re.sub(r'^[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)
    # collapse excess blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _clean_for_tts(text):
    """Clean Reddit-specific text patterns so TTS reads naturally."""
    # Reddit abbreviations → spoken equivalents
    abbrevs = [
        (r'\bAITA\b',   'Am I the asshole'),
        (r'\bAITAH\b',  'Am I the asshole'),
        (r'\bWIBTA\b',  'Would I be the asshole'),
        (r'\bNTA\b',    'not the asshole'),
        (r'\bYTA\b',    'you are the asshole'),
        (r'\bESH\b',    'everyone sucks here'),
        (r'\bNAH\b',    'no assholes here'),
        (r'\bINFO\b',   'I need more information'),
        (r'\bTIFU\b',   'today I messed up'),
        (r'\bTW\b',     ''),        # trigger warning label — just remove
        (r'\bCW\b',     ''),        # content warning label — just remove
        (r'\bOP\b',     'the original poster'),
        (r'\bSO\b',     'my partner'),
        (r'\bDH\b',     'my husband'),
        (r'\bDW\b',     'my wife'),
        (r'\bDS\b',     'my son'),
        (r'\bDD\b',     'my daughter'),
        (r'\bMIL\b',    'mother in law'),
        (r'\bFIL\b',    'father in law'),
        (r'\bSIL\b',    'sister in law'),
        (r'\bBIL\b',    'brother in law'),
        (r'\bNC\b',     'no contact'),
        (r'\bLC\b',     'low contact'),
        (r'\bLDR\b',    'long distance relationship'),
        (r'\bDM\b',     'message'),
        (r'\bIRL\b',    'in real life'),
        (r'\bIMO\b',    'in my opinion'),
        (r'\bIMHO\b',   'in my honest opinion'),
        (r'\bTBH\b',    'to be honest'),
        (r'\bNGL\b',    'not gonna lie'),
        (r'\bIDK\b',    'I do not know'),
        (r'\bIDC\b',    'I do not care'),
        (r'\bOMG\b',    'oh my god'),
        (r'\bWTF\b',    'what the heck'),
        (r'\bBS\b',     'nonsense'),
        (r'\bFYI\b',    'for your information'),
    ]
    for pattern, replacement in abbrevs:
        text = re.sub(pattern, replacement, text)

    # u/username and r/subreddit → remove
    text = re.sub(r'\bu/\w+', '', text)
    text = re.sub(r'\br/\w+', '', text)

    # Slashes between words → "or" (e.g. "he/she" → "he or she")
    text = re.sub(r'(\w+)/(\w+)', r'\1 or \2', text)

    # Numbers with k/m suffixes → spoken form (e.g. "5k" → "5 thousand")
    text = re.sub(r'\b(\d+)k\b', r'\1 thousand', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(\d+)m\b', r'\1 million', text, flags=re.IGNORECASE)

    # Clean up leftover punctuation and whitespace
    text = re.sub(r'\s{2,}', ' ', text)
    text = re.sub(r'\s([,.])', r'\1', text)
    return text.strip()


def _virality_score(p):
    """
    Score a Reddit post dict for viral potential.
    Higher = more likely to make an engaging video.
    """
    import math
    score = 0

    votes = max(p.get('score', 0), 1)
    num_comments = p.get('num_comments', 0)

    # High upvotes are good, but use log scale so a 200k post doesn't drown everything
    score += min(40, math.log10(votes) * 10)

    # Comment-to-upvote ratio — high ratio means divisive/discussion-heavy (more engaging)
    engagement = num_comments / votes
    score += min(25, engagement * 150)

    # Raw comment count also matters
    score += min(15, math.log10(max(num_comments, 1)) * 6)

    # Upvote ratio sweet spot: slightly controversial (0.65–0.85) drives more discussion
    ratio = p.get('upvote_ratio', 0.5)
    if 0.65 <= ratio <= 0.85:
        score += 15
    elif 0.85 < ratio <= 0.95:
        score += 7

    # Community awards are a strong signal of standout content
    awards = p.get('total_awards_received', 0)
    score += min(20, awards * 4)

    # Body length sweet spot for video (600–2500 chars reads well as TTS)
    body_len = len(p.get('selftext', ''))
    if 600 <= body_len <= 2500:
        score += 12
    elif 300 <= body_len < 600 or 2500 < body_len <= 4000:
        score += 5

    # Title emotional / drama keywords
    title = p.get('title', '').lower()
    body  = p.get('selftext', '').lower()

    viral_keywords = [
        'aita', 'tifu', 'found out', 'just found', 'caught', 'cheating',
        'divorce', 'betrayed', 'fired', 'quit my job', 'destroyed', 'ruined',
        'shocking', 'finally', 'update', 'lost everything', 'crying', 'cut off',
        'disowned', 'exposed', 'lied', 'secret', 'confess', 'truth came out',
        'worst day', 'insane', 'wild', 'can\'t believe', 'never told anyone',
        'years ago', 'years later', 'turns out', 'not the father', 'affair',
        'walked in on', 'found messages', 'found texts',
    ]
    keyword_hits = sum(1 for kw in viral_keywords if kw in title)
    score += min(20, keyword_hits * 6)

    # Hard penalty for explicitly sexual content — keeps videos YouTube-safe.
    # These posts exist on some NSFW-tagged subs but aren't suitable for the channel.
    explicit_keywords = [
        'sex tape', 'nude', 'nudes', 'onlyfans', 'porn', 'masturbat',
        'penis', 'vagina', 'blowjob', 'handjob', 'fingered', 'cum ',
        'cumshot', 'fetish', 'bdsm', 'kink', 'threesome', 'orgy',
    ]
    if any(kw in title or kw in body for kw in explicit_keywords):
        score -= 1000   # effectively removes it from contention

    return score


def _is_bot(author):
    """Return True if the Reddit username looks like a bot account."""
    if not author or author in ('[deleted]', '[removed]'):
        return True
    name = author.lower()
    # Common suffix/prefix patterns
    if name.endswith('bot') or name.startswith('bot_') or '_bot' in name:
        return True
    # Well-known bots that don't follow naming conventions
    known_bots = {
        'automoderator', 'remindmebot', 'repostsleuthbot', 'sneakpeek_bot',
        'totesmessenger', 'redditmetis', 'gifv-bot', 'anti-gif-bot',
        'vreddit_bot', 'converter-bot', 'tippr', 'reddit-stream',
        'ifttt', 'tweetposter', 'twitterbot', 'tweet_poster',
        'savevideo', 'stabbot', 'gifendgif', 'gifyoutube',
        'wikipedia_answer_bot', 'sub_analyzer_bot',
    }
    return name in known_bots


USED_POSTS_FILE = Path(__file__).parent / "used_posts.json"


def _load_used_posts():
    if USED_POSTS_FILE.exists():
        import json as _json
        return set(_json.loads(USED_POSTS_FILE.read_text()))
    return set()


def _save_used_post(post_id):
    import json as _json
    used = _load_used_posts()
    used.add(post_id)
    USED_POSTS_FILE.write_text(_json.dumps(list(used), indent=2))


def fetch_reddit_story(subreddit):
    """
    Pull a real post from Reddit's public JSON API — no API key needed.
    Uses the .json suffix that Reddit exposes on every listing and thread.
    Picks the first suitable text post from top/week, then grabs top 3 comments.
    """
    import urllib.request
    import urllib.error
    import json as _json
    import time

    sub     = subreddit.strip().lstrip('r').lstrip('/')
    headers = {'User-Agent': 'reddit-video-maker/1.0'}

    def _get(url):
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                return _json.loads(r.read())
        except urllib.error.HTTPError as e:
            print(f"❌ Reddit returned HTTP {e.code} — subreddit may not exist or is private.")
            sys.exit(1)
        except urllib.error.URLError as e:
            print(f"❌ Network error: {e.reason}")
            sys.exit(1)

    print(f"🌐 Fetching top posts from r/{sub} (all time)...")
    data  = _get(f"https://www.reddit.com/r/{sub}/top.json?t=all&limit=50&raw_json=1&include_over_18=1")
    posts = [p['data'] for p in data['data']['children'] if p['kind'] == 't3']

    used_posts = _load_used_posts()

    all_candidates = []
    candidates = []
    for p in posts:
        body = p.get('selftext', '').strip()
        if (p.get('is_self')
                and len(body) >= 300
                and body not in ('[removed]', '[deleted]', '')
                and p.get('num_comments', 0) >= 5
                and not _is_bot(p.get('author'))
                and p.get('distinguished') != 'moderator'):
            all_candidates.append(p)
            if p.get('id') not in used_posts:
                candidates.append(p)

    if not candidates:
        if not all_candidates:
            # Nothing usable at all — let caller try a different subreddit
            print(f"⚠️  No suitable posts found in r/{sub} — trying another subreddit...")
            return None
        # All good posts already used — settle for the best one anyway
        print(f"⚠️  All top posts in r/{sub} already used — reusing best available post.")
        candidates = all_candidates

    # Score all candidates and pick the highest-scoring one
    candidates.sort(key=_virality_score, reverse=True)
    chosen = candidates[0]

    if len(candidates) > 1:
        top_score = _virality_score(chosen)
        print(f"   📊 Scored {len(candidates)} candidates — top score: {top_score:.1f}")

    _save_used_post(chosen['id'])

    title    = chosen['title']
    body     = _clean_markdown(chosen['selftext'])
    author   = f"u/{chosen['author']}"
    upvotes  = _fmt_score(chosen['score'])
    sub_name = f"r/{chosen['subreddit']}"
    post_id  = chosen['id']

    print(f"   ✅ \"{title[:70]}{'...' if len(title) > 70 else ''}\"")
    print(f"      {upvotes} upvotes  |  {chosen['num_comments']:,} comments")

    time.sleep(0.5)   # be polite to Reddit's servers

    print("   Fetching comments...")
    cdata = _get(
        f"https://www.reddit.com/r/{sub}/comments/{post_id}.json"
        f"?limit=25&raw_json=1&sort=top"
    )

    comments = []
    for c in cdata[1]['data']['children']:
        if c['kind'] != 't1':
            continue
        cd = c['data']
        if _is_bot(cd.get('author')):
            continue
        if cd.get('distinguished') in ('moderator', 'admin'):
            continue
        text = _clean_markdown(cd.get('body', ''))
        if len(text) < 40 or text in ('[removed]', '[deleted]'):
            continue
        comments.append({
            'author':  f"u/{cd['author']}",
            'upvotes': _fmt_score(cd.get('score', 0)),
            'text':    text,
        })
        if len(comments) == 3:
            break

    if not comments:
        print("⚠️  No comments could be fetched — video will be post-only.")

    return {
        'subreddit':       sub_name,
        'title':           title,
        'author':          author,
        'upvotes':         upvotes,
        'narrator_gender': 'auto',
        'body':            body,
        'comments':        comments,
    }


def concatenate_videos(segment_paths, output_path):
    list_file = OUTPUT_DIR / "segments.txt"
    with open(list_file, "w") as f:
        for p in segment_paths:
            f.write(f"file '{p.resolve()}'\n")
    subprocess.run([
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(list_file),
        "-c", "copy",
        str(output_path)
    ], check=True)
    print(f"\n🎬 Final video saved: {output_path}")


def make_one_video(story, output_file, available_bgs, pipeline, whisper_model, test_mode):
    """
    Produce a single complete video from a story dict.

    Pipeline optimisation: while ffmpeg renders segment N (GPU via VideoToolbox),
    TTS + Whisper for segment N+1 runs concurrently on the CPU, cutting total
    time by roughly one full prepare-step per video.
    """
    global STORY
    STORY = story

    print(f"📁 Output: {output_file}")
    voice = detect_gender(story)

    # ── Generate all screenshots upfront (fast PIL, no reason to defer) ──
    print("\n📸 Generating screenshots...")
    screenshots, texts, labels = [], [], []

    screenshots.append(make_post_screenshot())
    texts.append(_clean_for_tts(f"{story['title']}. {story['body']}"))
    labels.append("post")

    # Comments removed — post only for YouTube Shorts format

    # ── Prepare one segment: TTS → Whisper → subtitle frames ─────────
    def _prepare(idx):
        label, text = labels[idx], texts[idx]
        if test_mode:
            audio    = generate_dummy_audio(5, f"{label}.wav")
            timings  = make_dummy_timings(text, 5)
        else:
            print(f"\n🎙️  TTS: {label}...")
            audio = generate_tts(text, f"{label}.wav", voice, pipeline)
            print(f"   📝 Transcribing {label}...")
            timings = transcribe_with_whisper(audio, whisper_model)
        return audio, create_subtitle_frames(timings, label)

    # ── Pipelined render loop ─────────────────────────────────────────
    # Kick off prep for segment 0 immediately, then for each segment:
    #   • wait for its prep result
    #   • launch prep for the NEXT segment in the background
    #   • render the current segment (ffmpeg, releases GIL → CPU is free for prep)
    segments = []
    n = 1 if test_mode else len(screenshots)

    with ThreadPoolExecutor(max_workers=2) as ex:
        pending = ex.submit(_prepare, 0)

        for idx in range(n):
            audio, sub_frames = pending.result()

            if idx + 1 < n:
                pending = ex.submit(_prepare, idx + 1)

            seg_path = OUTPUT_DIR / f"seg_{labels[idx]}.mp4"
            print(f"\n🎬 Rendering {labels[idx]} segment...")
            make_segment_video(screenshots[idx], audio, sub_frames, seg_path,
                               random.choice(available_bgs))
            segments.append(seg_path)

    if test_mode:
        concatenate_videos(segments, output_file)
        print(f"\n✅ Test done! Open '{output_file}' to check subtitles and layout.")
        print("   If it looks good, run normally: python3.10 reddit_video_maker.py")
        return

    # ── Combine ───────────────────────────────────────────────────────
    print("\n✂️  Concatenating all segments...")
    concatenate_videos(segments, output_file)
    print(f"\n✅ Done! Open '{output_file}' to watch your video.")


def upload_to_drive(file_path):
    """
    Upload a local file to Google Drive.
    Uses OAuth2 — on the first run it will open a browser to authorise access.
    The token is saved to token.json so subsequent runs are silent.

    Requirements:
        pip3.10 install google-api-python-client google-auth-httplib2 google-auth-oauthlib
        Place credentials.json (OAuth Desktop app) in the same directory as this script.
    """
    try:
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from google.auth.transport.requests import Request
    except ImportError:
        print("⚠️  Google Drive upload skipped — missing packages.")
        print("   Run: pip3.10 install google-api-python-client google-auth-httplib2 google-auth-oauthlib")
        return

    SCOPES = ['https://www.googleapis.com/auth/drive.file']
    creds_file = Path(__file__).parent / "credentials.json"
    token_file = Path(__file__).parent / "token.json"

    if not creds_file.exists():
        print("⚠️  Google Drive upload skipped — credentials.json not found.")
        print("   Create an OAuth Desktop app in Google Cloud Console, enable the Drive API,")
        print(f"   and download credentials.json to: {creds_file}")
        return

    creds = None
    if token_file.exists():
        creds = Credentials.from_authorized_user_file(str(token_file), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(str(creds_file), SCOPES)
            creds = flow.run_local_server(port=0)
        token_file.write_text(creds.to_json())

    service = build('drive', 'v3', credentials=creds)

    file_path = Path(file_path)
    metadata = {'name': file_path.name}
    if GOOGLE_DRIVE_FOLDER_ID:
        metadata['parents'] = [GOOGLE_DRIVE_FOLDER_ID]

    media = MediaFileUpload(str(file_path), mimetype='video/mp4', resumable=True)

    print(f"☁️  Uploading {file_path.name} to Google Drive...")
    uploaded = service.files().create(
        body=metadata,
        media_body=media,
        fields='id,name,webViewLink'
    ).execute()

    print(f"✅ Uploaded to Drive: {uploaded.get('name')}")
    print(f"   Link: {uploaded.get('webViewLink')}")


def _generate_youtube_metadata(story):
    """Auto-generate viral-optimised title, description, and tags from the story dict."""
    raw_title = story.get('title', 'Reddit Story')
    subreddit = story.get('subreddit', 'r/reddit')
    upvotes   = story.get('upvotes', '0')
    author    = story.get('author', 'u/unknown')
    body      = story.get('body', '').lower()
    sub_clean = subreddit.lstrip('r/').lower()

    # ── Title ────────────────────────────────────────────────────────────
    # Reformat Reddit-style prefixes into punchy hooks
    t = raw_title.strip()
    t = re.sub(r'\(\d+[MFmf]\)', '', t).strip()                         # remove (29F) tags
    t = re.sub(r'^AITA\s+for\s+', 'Was I Wrong For ', t, flags=re.IGNORECASE)
    t = re.sub(r'^AITA\b\s*[-:]*\s*', '', t, flags=re.IGNORECASE)
    t = re.sub(r'^AITAH?\s+for\s+', 'Was I Wrong For ', t, flags=re.IGNORECASE)
    t = re.sub(r'^TIFU\s+by\s+', 'I Messed Up By ', t, flags=re.IGNORECASE)
    t = re.sub(r'^TIFU\b\s*[-:]*\s*', '', t, flags=re.IGNORECASE)
    t = re.sub(r'^UPDATE\s*[-:]*\s*', 'UPDATE: ', t, flags=re.IGNORECASE)
    t = t[:77] + "..." if len(t) > 80 else t   # trim before adding suffix
    title = t.strip() + " #Shorts"

    # ── Tags ─────────────────────────────────────────────────────────────
    # Base viral tags — high search volume for this content type
    tags = [
        # Format / platform
        "shorts", "youtubeshorts", "viral", "fyp", "foryou", "foryoupage",
        "trending", "viralshorts", "viralvideo",
        # Reddit story niche
        "reddit", "redditstories", "reddittales", "redditreadings",
        "redditnarration", "redditdrama", "askreddit", "redditshorts",
        "bestofreddit", "redditreadit",
        # Story format
        "storytime", "storytelling", "storyshorts", "storytimeshorts",
        # Emotion hooks that drive clicks
        "unbelievable", "shocking", "crazy", "insane", "satisfying",
        "heartbreaking", "revenge",
        # Subreddit
        sub_clean, f"r {sub_clean}",
    ]

    # Subreddit-specific high-value tags
    sub_tag_map = {
        "amitheasshole":       ["aita", "am i the asshole", "aita reddit", "amitheasshole"],
        "aitah":               ["aita", "am i the asshole", "aita reddit"],
        "amiovereacting":      ["am i overreacting", "relationship drama"],
        "tifu":                ["tifu", "today i messed up", "embarrassing stories"],
        "relationship_advice": ["relationship advice", "relationship drama", "dating advice",
                                "relationship story", "dating stories"],
        "dating_advice":       ["dating advice", "dating stories", "dating drama"],
        "survivinginfidelity": ["cheating story", "infidelity", "betrayal", "cheating spouse"],
        "infidelity":          ["cheating story", "infidelity", "betrayal"],
        "deadbedrooms":        ["relationship drama", "marriage problems"],
        "entitledparents":     ["entitled parents", "karen", "entitled people", "karen stories"],
        "raisedbynarcissists": ["narcissistic parents", "toxic family", "family drama"],
        "justnomil":           ["mother in law", "mil stories", "family drama", "toxic family"],
        "justnofamily":        ["toxic family", "family drama", "family stories"],
        "weddingshaming":      ["wedding drama", "bridezilla", "wedding stories"],
        "pettyrevenge":        ["petty revenge", "revenge stories", "satisfying revenge"],
        "prorevenge":          ["pro revenge", "revenge stories", "satisfying revenge"],
        "nuclearrevenge":      ["nuclear revenge", "revenge stories", "satisfying"],
        "maliciouscompliance": ["malicious compliance", "satisfying stories", "revenge"],
        "traumatizethemback":  ["standing up for yourself", "revenge", "satisfying"],
        "choosingbeggars":     ["choosing beggars", "entitled people", "karen"],
        "antiwork":            ["antiwork", "quitting job", "work drama", "fired story",
                                "work stories", "quit my job"],
        "legaladvice":         ["legal advice", "law", "legal drama", "court stories"],
        "idontworkherelady":   ["work stories", "funny stories", "customer stories"],
        "talesfromretail":     ["retail stories", "customer stories", "work drama"],
        "confession":          ["confession", "confessions", "dark secrets"],
        "offmychest":          ["off my chest", "confession", "true stories"],
        "trueoffmychest":      ["true confession", "off my chest", "true stories"],
    }
    tags += sub_tag_map.get(sub_clean, [])

    # Content-based bonus tags detected from body/title
    combined = (raw_title + " " + body).lower()
    if any(w in combined for w in ["cheat", "affair", "infidel", "betray"]):
        tags += ["cheating story", "infidelity story", "betrayal"]
    if any(w in combined for w in ["divorce", "separat", "break up", "broke up"]):
        tags += ["divorce story", "breakup story", "relationship drama"]
    if any(w in combined for w in ["mother", "father", "parent", "family"]):
        tags += ["family drama", "family story", "toxic parents"]
    if any(w in combined for w in ["fired", "quit", "boss", "coworker", "job"]):
        tags += ["work drama", "job story", "workplace story"]
    if any(w in combined for w in ["revenge", "exposed", "karma"]):
        tags += ["karma", "satisfying", "justice served"]
    if any(w in combined for w in ["wedding", "bride", "groom", "married"]):
        tags += ["wedding drama", "wedding story", "marriage story"]

    # Deduplicate, strip invalid chars, enforce YouTube's 500 char total limit
    # YouTube only allows letters, numbers, spaces and hyphens in tags
    seen, unique_tags, total_chars = set(), [], 0
    for tag in tags:
        clean = re.sub(r'[^a-zA-Z0-9 \-]', '', tag).strip()
        if not clean or clean.lower() in seen:
            continue
        if len(clean) > 30:   # skip overly long individual tags
            continue
        if total_chars + len(clean) + 1 > 490:
            break
        seen.add(clean.lower())
        unique_tags.append(clean)
        total_chars += len(clean) + 1

    # ── Description ──────────────────────────────────────────────────────
    hashtags = (
        f"#reddit #redditstories #shorts #youtubeshorts #storytime "
        f"#viral #fyp #{sub_clean.replace('_', '')} #redditdrama"
    )
    description = (
        f"{raw_title}\n\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"From {subreddit} · {upvotes} upvotes\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"Subscribe for daily Reddit stories that will leave you speechless!\n"
        f"Turn on notifications so you never miss a story.\n\n"
        f"{hashtags}"
    )

    print(f"   📋 YouTube title: {title}")
    print(f"   🏷️  Tags ({len(unique_tags)}): {unique_tags}")
    return title, description, unique_tags


def upload_to_youtube(file_path, story):
    """
    Upload a rendered video to YouTube with auto-generated metadata.
    Uses OAuth2 — opens a browser on the first run, then saves youtube_token.json.

    Requirements:
        Same credentials.json used for Drive (enable YouTube Data API v3 in
        Google Cloud Console for the same project).
    """
    try:
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from google.auth.transport.requests import Request
    except ImportError:
        print("⚠️  YouTube upload skipped — missing packages.")
        print("   Run: pip3.10 install google-api-python-client google-auth-httplib2 google-auth-oauthlib")
        return

    SCOPES     = ["https://www.googleapis.com/auth/youtube.upload"]
    creds_file = Path(__file__).parent / "credentials.json"
    token_file = Path(__file__).parent / "youtube_token.json"

    if not creds_file.exists():
        print("⚠️  YouTube upload skipped — credentials.json not found.")
        return

    creds = None
    if token_file.exists():
        creds = Credentials.from_authorized_user_file(str(token_file), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow  = InstalledAppFlow.from_client_secrets_file(str(creds_file), SCOPES)
            creds = flow.run_local_server(port=0)
        token_file.write_text(creds.to_json())

    service = build("youtube", "v3", credentials=creds)

    title, description, tags = _generate_youtube_metadata(story)
    file_path = Path(file_path)

    body = {
        "snippet": {
            "title":       title,
            "description": description,
            "tags":        tags,
            "categoryId":  YOUTUBE_CATEGORY_ID,
        },
        "status": {
            "privacyStatus":           YOUTUBE_PRIVACY,
            "selfDeclaredMadeForKids": False,
        },
    }

    media = MediaFileUpload(str(file_path), mimetype="video/mp4",
                            chunksize=4 * 1024 * 1024, resumable=True)

    print(f"▶️  Uploading to YouTube: \"{title}\"")
    request = service.videos().insert(part="snippet,status", body=body, media_body=media)

    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            print(f"   ⏳ YouTube upload: {int(status.progress() * 100)}%")

    video_id = response.get("id")
    print(f"✅ YouTube upload complete!")
    print(f"   Link: https://www.youtube.com/watch?v={video_id}")


def main():
    TEST_MODE = "--test" in sys.argv

    # Pre-vetted subreddits: high drama/story content, safe for YouTube.
    # NSFW-tagged subs are included where the tag covers sensitive topics
    # (infidelity, trauma, family conflict) rather than explicit sexual content.
    # A keyword blocklist in _virality_score filters out explicitly sexual posts.
    RANDOM_SUBREDDITS = [
        # ── Core AITA / judgement ─────────────────────────────────────
        "AmItheAsshole",        # the OG viral drama machine
        "AITAH",                # sister sub, less strict rules = rawer stories
        "AmIOverreacting",      # newer, fast-growing, great engagement

        # ── Confessions & venting ─────────────────────────────────────
        "confession",
        "offmychest",
        "TrueOffMyChest",
        "tifu",                 # Today I F'd Up — high comment engagement

        # ── Relationships & infidelity ────────────────────────────────
        "relationship_advice",
        "dating_advice",
        "survivinginfidelity",  # NSFW-tagged but story-driven, very emotional
        "Infidelity",           # similar, raw first-person accounts
        "DeadBedrooms",         # relationship breakdown stories, huge sub

        # ── Family drama ──────────────────────────────────────────────
        "entitledparents",
        "raisedbynarcissists",
        "JUSTNOMIL",            # mother-in-law horror stories, extremely viral
        "JUSTNOFAMILY",         # broader toxic family stories
        "weddingshaming",       # bridezilla / family blow-ups at weddings

        # ── Revenge & justice ─────────────────────────────────────────
        "pettyrevenge",
        "ProRevenge",
        "NuclearRevenge",
        "MaliciousCompliance",
        "traumatizeThemBack",   # standing up to bullies, satisfying endings
        "ChoosingBeggars",      # entitled people getting shut down

        # ── Work & money ──────────────────────────────────────────────
        "antiwork",             # quitting/firing stories go viral constantly
        "legaladvice",          # real-stakes situations, high comment counts
        "IDontWorkHereLady",    # mistaken-employee stories, light but viral
        "TalesFromRetail",      # customer horror stories
    ]

    # Parse --count flag
    COUNT = 1
    for i, arg in enumerate(sys.argv):
        if arg == '--count' and i + 1 < len(sys.argv):
            try:
                COUNT = int(sys.argv[i + 1])
                if COUNT < 1:
                    raise ValueError
            except ValueError:
                print("❌ --count must be a positive integer  e.g. --count 5")
                sys.exit(1)
            break

    # Parse --reddit / --random flags
    USE_RANDOM = "--random" in sys.argv
    REDDIT_SUB = None
    if not USE_RANDOM:
        for i, arg in enumerate(sys.argv):
            if arg == '--reddit' and i + 1 < len(sys.argv):
                REDDIT_SUB = sys.argv[i + 1]
                break

    if TEST_MODE:
        print(f"\n🧪 TEST MODE — TTS skipped (silent audio)\n")
    else:
        label = f"{COUNT} video{'s' if COUNT > 1 else ''}"
        print(f"\n🚀 Reddit Video Maker — making {label}\n")

    available_bgs = [v for v in BACKGROUND_VIDEOS if os.path.exists(v)]
    missing_bgs   = [v for v in BACKGROUND_VIDEOS if not os.path.exists(v)]
    if missing_bgs:
        print(f"⚠️  Skipping missing background video(s): {', '.join(missing_bgs)}")
    if not available_bgs:
        print("❌ No background videos found. Add at least one file to BACKGROUND_VIDEOS.")
        sys.exit(1)
    print(f"🎬 Background pool: {len(available_bgs)} video(s)")

    if not TEST_MODE:
        try:
            import whisper
        except ImportError:
            print("❌ Whisper not installed. Run: pip3.10 install openai-whisper")
            sys.exit(1)

    setup_dirs()

    # ── Load models once (reused across all videos) ───────────────────
    pipeline      = None
    whisper_model = None
    if not TEST_MODE:
        from kokoro import KPipeline
        print("🔧 Loading Kokoro TTS model...")
        pipeline = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M')
        print("✅ Kokoro ready")

        print("🔧 Loading Whisper model (once for all videos)...")
        try:
            from faster_whisper import WhisperModel
            whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
            print(f"✅ faster-whisper ready ({WHISPER_MODEL_SIZE}, int8)")
        except ImportError:
            import whisper
            whisper_model = whisper.load_model(WHISPER_MODEL_SIZE)
            print(f"✅ Whisper ready ({WHISPER_MODEL_SIZE})  (tip: pip install faster-whisper for a free 4x speedup)")

    print(f"🖥️  Video encoder: {VIDEO_ENCODER}")

    # ── Main loop ─────────────────────────────────────────────────────
    # Drive uploads run in a background thread so the next video starts
    # rendering immediately rather than waiting for the upload to finish.
    upload_future = None

    with ThreadPoolExecutor(max_workers=1) as upload_ex:
        for video_num in range(COUNT):
            if COUNT > 1:
                print(f"\n{'═' * 54}")
                print(f"  VIDEO {video_num + 1} / {COUNT}")
                print(f"{'═' * 54}")

            if USE_RANDOM:
                # Keep cycling through shuffled subreddits until one works.
                # Loops forever — will always eventually find something.
                story = None
                while not story:
                    for sub in random.sample(RANDOM_SUBREDDITS, len(RANDOM_SUBREDDITS)):
                        print(f"🎲 Trying subreddit: r/{sub}")
                        story = fetch_reddit_story(sub)
                        if story:
                            break
                    if not story:
                        print("⚠️  All subreddits exhausted — retrying from the top...")
            elif REDDIT_SUB:
                story = None
                while not story:
                    story = fetch_reddit_story(REDDIT_SUB)
                    if not story:
                        print(f"⚠️  No suitable posts in r/{REDDIT_SUB} — retrying...")
                        time.sleep(5)
            else:
                story = STORY

            output_file = VIDS_DIR / "test_output.mp4" if TEST_MODE else get_next_video_number()
            make_one_video(story, output_file, available_bgs, pipeline, whisper_model, TEST_MODE)

            if not TEST_MODE:
                # Wait for any previous upload to finish before starting a new one
                if upload_future:
                    try:
                        upload_future.result()
                    except Exception as e:
                        print(f"⚠️  Previous upload failed: {e}")

                def _upload_all(path, s=story):
                    try:
                        # upload_to_drive(path)  # disabled
                        upload_to_youtube(path, s)
                    except Exception as e:
                        print(f"⚠️  Upload failed for {path}: {e}")

                print(f"   ☁️  Queuing upload for {Path(output_file).name}...")
                upload_future = upload_ex.submit(_upload_all, output_file)

        # Wait for the final upload to complete before exiting
        if upload_future:
            try:
                upload_future.result()
            except Exception as e:
                print(f"⚠️  Final upload failed: {e}")

    if COUNT > 1:
        print(f"\n🎉 All {COUNT} videos complete!")


if __name__ == "__main__":
    main()
