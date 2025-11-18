"""
Description:
- Module thu thập dữ liệu từ YouTube API
- Lấy thông tin playlist và video
- Lưu dữ liệu vào file JSON (metadata) và transcript (.txt)
"""

from googleapiclient.discovery import build
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)
from dotenv import load_dotenv
from typing import Optional, List, Dict, Any, Union, Tuple
from pathlib import Path
import yt_dlp
import os
import glob
import json
import re
import whisper
import subprocess

# --------------------------------------------------------
# Khởi tạo YouTube API client
# --------------------------------------------------------
load_dotenv()
youtube_api_key = os.environ["YOUTUBE_API_KEY"]
youtube = build("youtube", "v3", developerKey=youtube_api_key)


# --------------------------------------------------------
# Helper
# --------------------------------------------------------
def _resolve_device(device: Optional[str]) -> str:
    """
    'auto' → ưu tiên cuda > mps > cpu
    """
    device = (device or "auto").lower()
    if device != "auto":
        return device

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _fmt_ts(t: float) -> str:
    """
    Đổi giây -> H:MM:SS
    - Ví dụ:   3.5s  -> 0:00:03
               65s   -> 0:01:05
               3661s -> 1:01:01
    """
    t = max(0.0, float(t))
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    return f"{h}:{m:02d}:{s:02d}"


# --------------------------------------------------------
# Playlist metadata
# --------------------------------------------------------
class PlaylistMetadataFetch:
    def __init__(self, playlist_id: str):
        """Khởi tạo với playlist_id"""
        self.playlist_id = playlist_id
        self.playlist_name: Optional[str] = None
        self.playlist: Optional[Dict[str, Any]] = None
        self.videos: Optional[Dict[str, Any]] = None
        self.playlist_data: Optional[Dict[str, Any]] = None

    def fetch_playlist_info(self) -> Dict[str, Any]:
        """Lấy thông tin playlist"""
        try:
            self.playlist = (
                youtube.playlists().list(part="snippet", id=self.playlist_id).execute()
            )
            if not self.playlist.get("items"):
                raise ValueError(
                    f"Playlist {self.playlist_id} không tồn tại hoặc bị private"
                )
            return self.playlist
        except Exception as e:
            print(f"Lỗi khi lấy thông tin playlist {self.playlist_id}: {e}")
            raise

    def fetch_videos_info(self) -> Dict[str, Any]:
        """Lấy thông tin các videos trong playlist (hỗ trợ pagination)"""
        all_videos: List[Dict[str, Any]] = []
        next_page_token: Optional[str] = None

        try:
            while True:
                response = (
                    youtube.playlistItems()
                    .list(
                        part="snippet",
                        playlistId=self.playlist_id,
                        maxResults=50,
                        pageToken=next_page_token,
                    )
                    .execute()
                )

                all_videos.extend(response.get("items", []))
                next_page_token = response.get("nextPageToken")

                if not next_page_token:
                    break

            self.videos = {"items": all_videos}
            return self.videos

        except Exception as e:
            print(f"Lỗi khi lấy danh sách video: {e}")
            raise

    def get_important_info(self) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Trích các thông tin quan trọng từ playlist và video"""
        self.fetch_playlist_info()
        self.fetch_videos_info()
        playlist_data = self.playlist["items"][0]["snippet"]
        playlist_important_info = {
            "playlist_id": self.playlist_id,
            "title": playlist_data["title"],
            "description": playlist_data["description"],
            "published_at": playlist_data["publishedAt"],
            "channel_title": playlist_data["channelTitle"],
        }

        video_items = self.videos["items"]
        videos: List[Dict[str, Any]] = []
        for video in video_items:
            snippet = video["snippet"]
            video_id = snippet["resourceId"]["videoId"]

            # Bỏ qua video bị xóa hoặc private
            if video_id == "Deleted video" or snippet["title"] == "Private video":
                continue

            video_info = {
                "video_id": video_id,
                "title": snippet["title"],
                "url": f"https://www.youtube.com/watch?v={video_id}",
                "published_at": snippet["publishedAt"],
            }
            videos.append(video_info)

        return playlist_important_info, videos

    def convert_to_json_data(self) -> Dict[str, Any]:
        """Chuyển dữ liệu thành định dạng chuẩn để lưu vào file JSON"""
        playlist_info, videos = self.get_important_info()
        playlist_title = playlist_info["title"]

        # Tạo folder name an toàn
        folder_name = self._sanitize_folder_name(playlist_title)

        data = {**playlist_info, "total_videos": len(videos), "videos": videos}
        self.playlist_name = folder_name
        self.playlist_data = data
        return data

    @staticmethod
    def _sanitize_folder_name(name: str) -> str:
        """Tạo tên folder an toàn từ tên playlist"""
        # Loại bỏ ký tự đặc biệt
        name = re.sub(r'[<>:"/\\|?*]', "", name)
        # Chuyển thành lowercase và thay space bằng dash
        name = name.lower().strip().replace(" ", "-")
        # Loại bỏ dash liên tiếp
        name = re.sub(r"-+", "-", name)
        # Giới hạn độ dài
        return name[:100]


# --------------------------------------------------------
# Transcript bằng YouTubeTranscriptAPI
# --------------------------------------------------------
class TranscriptAPIFetcher:
    def __init__(self, preferred_langs: Optional[List[str]] = None):
        """
        Fetch transcript cho video youtube bằng YouTubeTranscriptAPI
        """
        self.preferred_langs = preferred_langs or ["vi", "en"]
        self.transcript = None

    def fetch_transcript_from(self, video_id: str) -> Optional[List[Dict[str, Any]]]:
        """Lấy transcript từ video_id"""
        api = YouTubeTranscriptApi()
        try:
            transcript_list = api.list(video_id)

            # Ưu tiên manually-created (phụ đề thủ công)
            for lang in self.preferred_langs:
                try:
                    transcript = transcript_list.find_manually_created_transcript(
                        [lang]
                    )
                    self.transcript = transcript
                    return transcript.fetch().to_raw_data()
                except NoTranscriptFound:
                    continue

            # Ưu tiên generated (phụ đề tự động)
            for lang in self.preferred_langs:
                try:
                    transcript = transcript_list.find_generated_transcript([lang])
                    self.transcript = transcript
                    return transcript.fetch().to_raw_data()
                except NoTranscriptFound:
                    continue

            # Fallback: lấy bất kỳ phụ đề nào
            try:
                transcript = transcript_list.find_transcript(self.preferred_langs)
                self.transcript = transcript
                return transcript.fetch().to_raw_data()
            except NoTranscriptFound:
                return None

        except (TranscriptsDisabled, VideoUnavailable) as e:
            print(f"Video {video_id} không có transcript: {e}")
            return None
        except Exception as e:
            print(f"Lỗi không xác định khi lấy transcript cho {video_id}: {e}")
            return None


# --------------------------------------------------------
# Transcript bằng Whisper (openai-whisper)
# --------------------------------------------------------
class TranscriptWhisperFetcher:
    def __init__(
        self,
        audio_dir: str = "audio",
        model_size: str = "medium",
        language: str = "vi",
        device: str = "auto",
        initial_prompt: Optional[str] = None,
    ):
        """
        Fetch transcript bằng openai-whisper (local)
        - Tải audio từ YouTube
        - Transcribe toàn bộ audio
        - Trả về text + segments (có timestamp)
        """
        self.audio_dir = Path(audio_dir)
        self.audio_dir.mkdir(exist_ok=True)

        self.model_size = model_size
        self.language = language
        self.device = _resolve_device(device)
        self.initial_prompt = initial_prompt or (
            "Chủ đề: học sâu, machine learning, logistic regression, CNN, ImageNet."
        )

        print(
            f"🔧 Load Whisper (openai-whisper) model='{self.model_size}' trên {self.device}..."
        )
        self.model = whisper.load_model(self.model_size, device=self.device)
        print("✅ Whisper model đã sẵn sàng!")

    def _decode_options(self) -> Dict[str, Any]:
        opts: Dict[str, Any] = {
            "verbose": False,
            "condition_on_previous_text": False,
        }
        if self.language:
            opts["language"] = self.language
        if self.initial_prompt:
            opts["initial_prompt"] = self.initial_prompt
        return opts

    def get_audio_from(self, video_id: str) -> Path:
        """
        Tải audio từ YouTube và convert sang .wav
        Trả về path file wav (trong self.audio_dir)
        """
        url = f"https://www.youtube.com/watch?v={video_id}"
        raw_outtmpl = str(self.audio_dir / f"{video_id}.%(ext)s")
        final_audio_path = self.audio_dir / f"{video_id}.wav"

        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": raw_outtmpl,
            "quiet": True,
            "no_warnings": True,
            "proxy": "",
        }

        print(f"📥 Tải audio từ YouTube: {url}")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        # lấy file tải về (bất kỳ đuôi nào, nhưng tránh .wav nếu đã tồn tại)
        raw_files = glob.glob(str(self.audio_dir / f"{video_id}.*"))
        if not raw_files:
            raise FileNotFoundError(
                f"Không tìm thấy file audio sau khi tải: {video_id}"
            )

        # Ưu tiên file KHÔNG phải .wav
        raw_path: Optional[Path] = None
        for p in raw_files:
            if not p.lower().endswith(".wav"):
                raw_path = Path(p)
                break
        if raw_path is None:
            # Nếu chỉ có .wav thì dùng luôn
            raw_path = Path(raw_files[0])

        # Nếu đã có wav cũ thì xoá trước cho chắc
        if final_audio_path.exists():
            try:
                final_audio_path.unlink()
            except Exception as e:
                print(f"⚠ Không xoá được file wav cũ: {final_audio_path} - {e}")

        # convert sang wav 16k mono bằng subprocess
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(raw_path),
            "-ar",
            "16000",
            "-ac",
            "1",
            str(final_audio_path),
        ]
        print("🎬 Chạy lệnh ffmpeg:", " ".join(cmd))
        proc = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore"
        )

        if proc.returncode != 0:
            print("❌ ffmpeg error:")
            print(proc.stderr)
            raise RuntimeError("ffmpeg failed khi convert audio sang wav")

        # Xoá file gốc nếu khác wav
        try:
            if raw_path.exists() and raw_path != final_audio_path:
                raw_path.unlink()
        except Exception as e:
            print(f"⚠ Không xoá được file audio gốc: {raw_path} - {e}")

        if not final_audio_path.exists():
            raise FileNotFoundError("Không tạo được file WAV cho Whisper")

        print(f"🎧 Audio đã convert: {final_audio_path}")
        return final_audio_path

    def transcribe_audio_from(
        self,
        audio_path: Union[str, Path],
        show_segments: bool = True,
    ) -> Dict[str, Any]:
        """
        Transcribe 1 file audio bằng Whisper (openai-whisper)

        Return:
            {
              "text": "<plain text (mỗi segment 1 dòng)>",
              "segments": [
                  {"start": float, "end": float, "text": str},
                  ...
              ]
            }
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            print(f"❌ Audio file không tồn tại: {audio_path}")
            return {"text": "", "segments": []}

        print(f"📝 Bắt đầu transcribe: {audio_path}")
        try:
            result = self.model.transcribe(str(audio_path), **self._decode_options())
        except Exception as e:
            print(f"❌ Lỗi khi transcribe {audio_path}: {e}")
            return {"text": "", "segments": []}

        raw_segments = result.get("segments") or []
        segments: List[Dict[str, Any]] = []

        if show_segments:
            print("\n========== SEGMENTS ==========\n")

        for idx, seg in enumerate(raw_segments):
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", start))
            text = (seg.get("text") or "").strip()
            if not text:
                continue

            if show_segments:
                print(f"[{idx + 1:03d}] {_fmt_ts(start)} - {_fmt_ts(end)} | {text}")

            segments.append(
                {
                    "start": start,
                    "end": end,
                    "text": text,
                }
            )

        if show_segments:
            print("\n========== DONE ==========\n")

        lines = [s["text"] for s in segments]
        full_text = "\n".join(lines).strip()

        return {
            "text": full_text,
            "segments": segments,
        }

    def fetch_transcript_from(
        self,
        video_id: str,
        cleanup: bool = True,
        show_segments: bool = True,
    ) -> Dict[str, Any]:
        """
        Full pipeline: tải audio → transcribe → (option) xóa audio tạm
        """
        audio_path: Optional[Path] = None
        try:
            audio_path = self.get_audio_from(video_id)
            data = self.transcribe_audio_from(audio_path, show_segments=show_segments)
            return data
        finally:
            if cleanup and audio_path is not None and audio_path.exists():
                try:
                    audio_path.unlink()
                    print(f"🧹 Đã xóa file audio tạm: {audio_path}")
                except Exception as e:
                    print(f"⚠ Không thể xóa file audio tạm: {e}")


# --------------------------------------------------------
# Utility functions
# --------------------------------------------------------
def extract_playlist_id(url: str) -> Optional[str]:
    """Trích xuất playlist ID từ URL"""
    patterns = [
        r"list=([a-zA-Z0-9_-]+)",
        r"playlist\?list=([a-zA-Z0-9_-]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    # Nếu đã là ID thuần
    if re.match(r"^[a-zA-Z0-9_-]+$", url):
        return url

    return None


def save_json(data: Dict[str, Any], file_path: Union[str, Path]) -> None:
    """Lưu dữ liệu vào file JSON"""
    try:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Đã lưu: {file_path}")
    except Exception as e:
        print(f"Lỗi khi lưu file {file_path}: {e}")
        raise


def load_json(file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """Đọc dữ liệu từ file JSON"""
    file_path = Path(file_path)
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"Lỗi khi đọc file {file_path}: {e}")
        return None


# --------------------------------------------------------
# Chuẩn hoá segments & format TXT
# --------------------------------------------------------
def normalize_api_segments(raw_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    API: [{'text':..., 'start':..., 'duration':...}, ...]
    → [{'start':..., 'end':..., 'text':...}, ...]
    """
    norm: List[Dict[str, Any]] = []
    for seg in raw_segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        start = float(seg.get("start", 0.0))
        dur = float(seg.get("duration", 0.0))
        end = start + max(0.0, dur)
        norm.append(
            {
                "start": start,
                "end": end,
                "text": text.replace("\n", " "),
            }
        )
    return norm


def normalize_whisper_segments(
    raw_segments: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Whisper: [{'start':..., 'end':..., 'text':...}]
    → format giống API: [{'start', 'end', 'text'}, ...]
    """
    norm: List[Dict[str, Any]] = []
    for seg in raw_segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        norm.append(
            {
                "start": float(seg.get("start", 0.0)),
                "end": float(seg.get("end", 0.0)),
                "text": text.replace("\n", " "),
            }
        )
    return norm


def segments_to_txt_with_timestamp(segments: List[Dict[str, Any]]) -> str:
    """
    Input:  [{'start':..., 'end':..., 'text':...}]
    Output: string nhiều dòng (mỗi line một segment):
      H:MM:SS - H:MM:SS, nội dung
    """
    lines: List[str] = []
    for seg in segments:
        s_ts = _fmt_ts(seg["start"])
        e_ts = _fmt_ts(seg["end"])
        text = seg["text"]
        lines.append(f"{s_ts} - {e_ts}, {text}")
    return "\n".join(lines)
