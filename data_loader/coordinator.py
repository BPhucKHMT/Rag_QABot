"""
File: coordinator.py
Chức năng:
- Điều phối pipeline thu thập dữ liệu YouTube cho một playlist
- Lấy playlist metadata
- Lấy transcript (ưu tiên API → fallback Whisper)
- Chuẩn hoá và lưu transcript vào .txt
- Cập nhật playlists_index.json để quản lý nhiều playlist
- [MỚI] Hỗ trợ đọc từ config.yaml để xử lý nhiều playlist tự động

Cấu trúc thư mục (tính từ root project, ví dụ: Rag_QABot/):
- Rag_QABot/
    - loader/
        - youtube_fetchers.py
        - coordinator.py   (file này)
    - config.yaml       (file config)
    - data/
        - playlists_index.json
        - logs/
        - <playlist_folder>/
            - metadata.json
            - transcripts/
                - <video_id>.txt
            - audio/        (audio tạm cho Whisper, file .wav sẽ bị xóa sau)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional
import json
import datetime
import argparse
import time
import yaml

# ================================================================
# Import từ youtube_fetchers.py
#   LƯU Ý: file này nằm cùng thư mục với coordinator.py
#   Và bạn CHẠY bằng: python -m loader.coordinator từ root repo
# ================================================================
from youtube_fetchers import (
    PlaylistMetadataFetch,
    TranscriptAPIFetcher,
    TranscriptWhisperFetcher,
    normalize_api_segments,
    normalize_whisper_segments,
    segments_to_txt_with_timestamp,
    extract_playlist_id,
    save_json,
)

# =====================================================================
# Đường dẫn: data/ nằm CÙNG CẤP với thư mục loader/
# =====================================================================
# __file__ = <root>/loader/coordinator.py
# parents[0] = <root>/loader
# parents[1] = <root>
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT_DIR / "data"
LOGS_DIR = DATA_ROOT / "logs"
INDEX_FILE = DATA_ROOT / "playlists_index.json"
CONFIG_FILE = ROOT_DIR / "config.yaml"


# =====================================================================
# Helper
# =====================================================================
def ensure_dirs() -> None:
    """Tạo thư mục data/ và data/logs/ nếu chưa có"""
    DATA_ROOT.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)


def load_index() -> Dict[str, Any]:
    """Đọc playlists_index.json (nếu chưa có thì trả về khung rỗng)"""
    if not INDEX_FILE.exists():
        return {"playlists": []}
    try:
        return json.loads(INDEX_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {"playlists": []}


def save_index(data: Dict[str, Any]) -> None:
    """Lưu playlists_index.json"""
    INDEX_FILE.parent.mkdir(exist_ok=True)
    INDEX_FILE.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"💾 Đã cập nhật index: {INDEX_FILE}")


def upsert_index(
    index: Dict[str, Any], playlist_info: Dict[str, Any], folder_name: str
) -> None:
    """
    Cập nhật hoặc thêm mới playlist trong playlists_index.json
    """
    pid = playlist_info["playlist_id"]
    playlists = index.setdefault("playlists", [])

    for item in playlists:
        if item["playlist_id"] == pid:
            item.update(
                {
                    "title": playlist_info.get("title"),
                    "folder_name": folder_name,
                    "total_videos": playlist_info.get("total_videos"),
                    "processed_videos": playlist_info.get("processed_videos"),
                    "failed_videos": playlist_info.get("failed_videos"),
                    "updated_at": datetime.datetime.now().isoformat(),
                }
            )
            return

    playlists.append(
        {
            "playlist_id": pid,
            "folder_name": folder_name,
            "title": playlist_info.get("title"),
            "total_videos": playlist_info.get("total_videos"),
            "processed_videos": playlist_info.get("processed_videos", 0),
            "failed_videos": playlist_info.get("failed_videos", 0),
            "created_at": datetime.datetime.now().isoformat(),
        }
    )


def save_txt(content: str, path: str | Path) -> str:
    """
    Lưu nội dung ra file .txt
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print(f"💾 Đã lưu transcript TXT: {path}")
    return str(path)


def is_youtube_block_error(err: Exception | str) -> bool:
    """
    Nhận diện lỗi YouTube block IP từ message
    """
    msg = str(err)
    return "YouTube is blocking requests from your IP" in msg


# =====================================================================
# COORDINATOR
# =====================================================================
class DataCoordinator:
    def __init__(self, sleep_between_videos: float = 2.0):
        """
        sleep_between_videos: thời gian nghỉ (giây) giữa 2 video
        để hạn chế bị YouTube block vì quá nhiều request.
        """
        ensure_dirs()
        self.index = load_index()
        self.api_fetcher = TranscriptAPIFetcher()
        self.sleep_between_videos = sleep_between_videos

    # ------------------------------------------------------------
    # Run pipeline cho 1 playlist
    # ------------------------------------------------------------
    def process_playlist(self, playlist_url_or_id: str, limit: Optional[int] = None):
        """
        playlist_url_or_id: URL hoặc ID playlist
        limit: nếu muốn giới hạn số video xử lý (vd: 10), còn None thì xử lý hết
        """
        pid = extract_playlist_id(playlist_url_or_id)
        if not pid:
            raise ValueError("Không tìm thấy playlist id hợp lệ!")

        print(f"🔎 Đang xử lý playlist: {pid}")
        meta_fetcher = PlaylistMetadataFetch(pid)
        playlist_data = meta_fetcher.convert_to_json_data()
        folder_name = meta_fetcher.playlist_name

        # Thư mục playlist: data/<playlist_folder>/
        playlist_folder = DATA_ROOT / folder_name
        playlist_folder.mkdir(parents=True, exist_ok=True)

        # Lưu metadata.json
        metadata_path = playlist_folder / "metadata.json"
        save_json(playlist_data, metadata_path)

        # Thư mục transcripts: data/<playlist_folder>/transcripts/
        transcripts_dir = playlist_folder / "transcripts"
        transcripts_dir.mkdir(exist_ok=True)

        # Thư mục audio riêng cho playlist: data/<playlist_folder>/audio/
        audio_dir = playlist_folder / "audio"
        audio_dir.mkdir(exist_ok=True)

        # Whisper fetcher dùng audio_dir của playlist này
        whisper_fetcher = TranscriptWhisperFetcher(audio_dir=str(audio_dir))

        # Ghi log: data/logs/<playlist_folder>.log
        log_file = LOGS_DIR / f"{folder_name}.log"
        log_f = log_file.open("a", encoding="utf-8")
        log_f.write(f"\n\n=== Process start {datetime.datetime.now()} ===\n")

        videos = playlist_data["videos"]
        if limit is not None:
            videos = videos[:limit]

        total_videos = len(videos)
        success_count = 0
        fail_count = 0

        # Flag để biết có bị YouTube block hay không
        blocked_by_youtube = False

        # ------------------------------------------------------------
        # Lặp qua từng video trong playlist
        # ------------------------------------------------------------
        for idx, vid in enumerate(videos, start=1):
            video_id = vid["video_id"]
            video_title = vid["title"]

            if blocked_by_youtube:
                msg = "⛔ Bỏ qua video vì đã bị YouTube block IP trước đó."
                print(msg)
                log_f.write(msg + "\n")
                fail_count += 1
                continue

            log_f.write(
                f"\n>>> VIDEO {idx}/{total_videos}: {video_id} | {video_title}\n"
            )
            print(f"\n🎬 VIDEO {idx}/{total_videos}: {video_id} | {video_title}")

            txt_path = transcripts_dir / f"{video_id}.txt"

            # Skip nếu transcript đã tồn tại
            if txt_path.exists():
                print("⏭ Transcript đã tồn tại, bỏ qua.")
                log_f.write("⏭ Skipped (exists)\n")
                success_count += 1
                time.sleep(self.sleep_between_videos)
                continue

            # --------------------------------------------------------
            # 1) Try API transcript
            # --------------------------------------------------------
            api_segments = None
            try:
                api_segments = self.api_fetcher.fetch_transcript_from(video_id)
            except Exception as e:
                print(f"⚠ Lỗi API transcript: {e}")
                log_f.write(f"⚠ API error: {e}\n")
                if is_youtube_block_error(e):
                    blocked_by_youtube = True

            if api_segments and not blocked_by_youtube:
                log_f.write("✓ API transcript OK\n")
                print("📄 Lấy transcript API thành công.")

                segments = normalize_api_segments(api_segments)
                txt = segments_to_txt_with_timestamp(segments)
                save_txt(txt, txt_path)
                success_count += 1
                time.sleep(self.sleep_between_videos)
                continue

            # --------------------------------------------------------
            # 2) Fallback → Whisper
            # --------------------------------------------------------
            if blocked_by_youtube:
                msg = "⛔ Bị YouTube block IP → không thử Whisper để tránh spam."
                print(msg)
                log_f.write(msg + "\n")
                fail_count += 1
                continue

            print("⚠ API transcript không có → dùng Whisper...")
            log_f.write("⚠ API failed → Whisper\n")

            try:
                whisper_data = whisper_fetcher.fetch_transcript_from(
                    video_id, cleanup=True, show_segments=True
                )
                segments = normalize_whisper_segments(whisper_data["segments"])
                txt = segments_to_txt_with_timestamp(segments)
                save_txt(txt, txt_path)
                success_count += 1
                log_f.write("✓ Whisper transcript OK\n")
            except Exception as e:
                err_msg = f"❌ Whisper failed: {e}"
                print(err_msg)
                log_f.write(err_msg + "\n")
                fail_count += 1
                if is_youtube_block_error(e):
                    blocked_by_youtube = True

            # nghỉ một chút sau mỗi video
            time.sleep(self.sleep_between_videos)

        # ------------------------------------------------------------
        # Cập nhật playlists_index.json
        # ------------------------------------------------------------
        playlist_data["processed_videos"] = success_count
        playlist_data["failed_videos"] = fail_count

        upsert_index(self.index, playlist_data, folder_name)
        save_index(self.index)

        log_f.write(
            f"\n=== Done {datetime.datetime.now()} | success={success_count}, fail={fail_count} ===\n"
        )
        log_f.close()

        print("\n🎉 Hoàn tất pipeline cho playlist!")
        print(f"📁 Lưu tại: {playlist_folder}")
        print(f"📄 Log: {log_file}")
        if blocked_by_youtube:
            print(
                "⛔ Lưu ý: Có dấu hiệu YouTube block IP, nên nghỉ một thời gian trước khi chạy lại."
            )


# =====================================================================
# CONFIG-BASED COORDINATOR
# =====================================================================
class ConfigBasedCoordinator:
    """
    Coordinator mới đọc từ config.yaml
    Tự động quét và xử lý playlist
    """

    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            config_path = str(CONFIG_FILE)

        self.config_path = Path(config_path)
        self.config = self._load_config()

        # Lấy settings
        settings = self.config.get("settings", {})
        sleep = settings.get("sleep_between_videos", 2.0)

        self.coordinator = DataCoordinator(sleep_between_videos=sleep)

    def _load_config(self) -> Dict[str, Any]:
        """Load config.yaml"""
        if not self.config_path.exists():
            print(f"⚠️ Không tìm thấy file config: {self.config_path}")
            print("   Tạo file config.yaml mẫu...")
            self._create_sample_config()
            return self._load_config()

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"❌ Lỗi đọc config: {e}")
            return {}

    def _create_sample_config(self):
        """Tạo file config.yaml mẫu"""
        sample = {
            "playlists": [
                {
                    "url": "https://www.youtube.com/playlist?list=PLxxxxxx",
                    "enabled": True,
                },
            ],
            "settings": {
                "sleep_between_videos": 2.0,
                "limit_per_playlist": None,
            },
        }
        CONFIG_FILE.parent.mkdir(exist_ok=True)
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            yaml.dump(sample, f, allow_unicode=True, default_flow_style=False)
        print(f"✅ Đã tạo config mẫu: {CONFIG_FILE}")

    def process_all_enabled_playlists(self):
        """Xử lý tất cả playlist enabled trong config"""
        playlists = self.config.get("playlists", [])
        if not playlists:
            print("⚠️ Không có playlist nào trong config")
            return

        enabled_playlists = [p for p in playlists if p.get("enabled", True)]

        if not enabled_playlists:
            print("⚠️ Không có playlist nào được enable")
            return

        print(f"📋 Tìm thấy {len(enabled_playlists)} playlist cần xử lý\n")

        for idx, playlist in enumerate(enabled_playlists, 1):
            url = playlist["url"]
            print(f"\n{'=' * 60}")
            print(f"📌 Playlist {idx}/{len(enabled_playlists)}")
            print(f"🔗 URL: {url}")
            print(f"{'=' * 60}\n")

            try:
                limit = self.config["settings"].get("limit_per_playlist")
                self.coordinator.process_playlist(url, limit=limit)
            except Exception as e:
                print(f"❌ Lỗi xử lý playlist: {e}")
                continue

        print("\n🎉 Hoàn tất xử lý tất cả playlist!")

    def add_playlist(self, url: str, enabled: bool = True):
        """Thêm playlist mới vào config"""
        playlists = self.config.setdefault("playlists", [])

        # Kiểm tra trùng
        if any(p["url"] == url for p in playlists):
            print("⚠️ Playlist đã tồn tại trong config")
            return False

        playlists.append({"url": url, "enabled": enabled})
        self._save_config()
        print(f"✅ Đã thêm playlist vào config: {url}")
        return True

    def _save_config(self):
        """Lưu config.yaml"""
        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.dump(self.config, f, allow_unicode=True, default_flow_style=False)


# =====================================================================
# CLI
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="YouTube Data Coordinator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  # Xử lý 1 playlist (chế độ cũ)
  python -m loader.coordinator --playlist "https://youtube.com/playlist?list=PLxxx"
  
  # Xử lý tất cả playlist từ config.yaml
  python -m loader.coordinator --config
  
  # Thêm playlist vào config
  python -m loader.coordinator --add-playlist "https://youtube.com/playlist?list=PLxxx"
        """,
    )

    # Chế độ cũ: xử lý 1 playlist
    parser.add_argument(
        "--playlist", type=str, help="Playlist URL hoặc ID (chế độ single playlist)"
    )

    # Chế độ mới: dùng config
    parser.add_argument("--config", action="store_true", help="Chạy theo config.yaml")

    parser.add_argument(
        "--add-playlist", type=str, help="Thêm playlist mới vào config.yaml"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Giới hạn số video xử lý (vd: 5). Mặc định: xử lý tất cả.",
    )

    parser.add_argument(
        "--sleep",
        type=float,
        default=2.0,
        help="Số giây nghỉ giữa 2 video để tránh bị YouTube block. Mặc định: 2.0",
    )

    args = parser.parse_args()

    if args.config:
        # Chế độ config-based
        config_coord = ConfigBasedCoordinator()
        config_coord.process_all_enabled_playlists()

    elif args.add_playlist:
        # Thêm playlist vào config
        config_coord = ConfigBasedCoordinator()
        config_coord.add_playlist(args.add_playlist)

    elif args.playlist:
        # Chế độ cũ: xử lý 1 playlist
        coord = DataCoordinator(sleep_between_videos=args.sleep)
        coord.process_playlist(args.playlist, limit=args.limit)

    else:
        parser.print_help()
