"""
Analyzes images, video and audio using the current Gemini multimodal SDK.
"""

from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MAX_AUDIO_SIZE_MB, MAX_VIDEO_FRAMES, MODEL_FAST, VIDEO_FRAME_INTERVAL_SEC
from llm import generate_multimodal, generate_text, upload_file, wait_for_file


def analyze_image(image_path: Path) -> str:
    """Describe an image in an educational context."""
    try:
        import PIL.Image

        img = PIL.Image.open(image_path)
        return generate_multimodal(
            [
                img,
                (
                    "Опишите это изображение в контексте учебного материала: что изображено, "
                    "какую образовательную ценность несёт. Ответ на русском языке, до 150 слов."
                ),
            ],
            model=MODEL_FAST,
        )
    except Exception as e:
        return f"Ошибка анализа изображения: {e}"


def analyze_video(video_path: Path) -> str:
    """
    Upload video to Gemini Files API and describe it.
    Falls back to frame-by-frame analysis if upload fails.
    """
    try:
        video_file = upload_file(video_path)
        video_file = wait_for_file(video_file.name, attempts=30, delay_sec=3)

        if getattr(getattr(video_file, "state", None), "name", "") == "FAILED":
            raise RuntimeError("Gemini video processing failed")

        return generate_multimodal(
            [
                video_file,
                (
                    "Опишите содержание этого видеоматериала в учебном контексте: "
                    "тему, ключевые моменты, образовательную ценность. "
                    "Ответ на русском, до 200 слов."
                ),
            ],
            model=MODEL_FAST,
        )
    except Exception:
        return _analyze_video_frames(video_path)


def _analyze_video_frames(video_path: Path) -> str:
    """Fall-back: extract frames with OpenCV and describe them."""
    try:
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        interval = max(1, int(fps * VIDEO_FRAME_INTERVAL_SEC))
        descriptions = []
        frame_idx = 0
        extracted = 0

        frames_dir = video_path.parent / f"{video_path.stem}_frames"
        frames_dir.mkdir(exist_ok=True)

        while cap.isOpened() and extracted < MAX_VIDEO_FRAMES:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % interval == 0:
                ts = frame_idx / fps
                fp = frames_dir / f"frame_{extracted:03d}_{ts:.1f}s.jpg"
                cv2.imwrite(str(fp), frame)
                descriptions.append(f"[{ts:.1f}с] {analyze_image(fp)}")
                extracted += 1
            frame_idx += 1

        cap.release()

        if not descriptions:
            return f"Видеофайл: {video_path.name} (анализ кадров не удался)"

        summary_prompt = (
            "Составьте краткое описание видеоурока на основе описаний кадров:\n"
            + "\n".join(descriptions)
            + "\nОтвет до 200 слов."
        )
        return generate_text(summary_prompt, model=MODEL_FAST)
    except Exception as e:
        return f"Ошибка анализа видео: {e}"


def analyze_audio(audio_path: Path) -> str:
    """Transcribe / describe audio using Gemini Files API."""
    try:
        file_size_mb = audio_path.stat().st_size / (1024 * 1024)
        mime_map = {
            ".mp3": "audio/mpeg",
            ".wav": "audio/wav",
            ".ogg": "audio/ogg",
            ".flac": "audio/flac",
            ".m4a": "audio/mp4",
            ".aac": "audio/aac",
            ".opus": "audio/ogg",
            ".oga": "audio/ogg",
        }
        mime = mime_map.get(audio_path.suffix.lower(), "audio/mpeg")

        if file_size_mb <= MAX_AUDIO_SIZE_MB:
            audio_file = upload_file(audio_path, mime_type=mime)
            audio_file = wait_for_file(audio_file.name, attempts=20, delay_sec=2)

            return generate_multimodal(
                [
                    audio_file,
                    (
                        "Транскрибируйте или опишите содержание данного аудиоматериала "
                        "в учебном контексте. Ответ на русском языке."
                    ),
                ],
                model=MODEL_FAST,
            )

        return (
            f"Аудиофайл: {audio_path.name} ({file_size_mb:.1f} МБ) "
            "— слишком большой для анализа."
        )
    except Exception as e:
        try:
            import whisper

            model = whisper.load_model("base")
            result = model.transcribe(str(audio_path))
            return "Транскрипция: " + result.get("text", "")[:2000]
        except ImportError:
            pass
        return f"Ошибка анализа аудио: {e}"
