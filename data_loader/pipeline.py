"""
File: pipeline.py
Chức năng: Chạy toàn bộ pipeline tự động
    1. Crawl transcript từ YouTube (coordinator.py)
    2. Preprocess transcript (preprocess.py) 
    3. Index vào Vector Database (index_data.py)

Cách dùng:
    python pipeline.py                    # Chạy full pipeline
    python pipeline.py --skip-crawl       # Bỏ qua crawl (nếu đã có transcript)
    python pipeline.py --skip-preprocess  # Bỏ qua preprocess
    python pipeline.py --only-index       # Chỉ index (bỏ qua 2 bước đầu)
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Import các module cần thiết
try:
    from data_loader.coordinator import ConfigBasedCoordinator
    from data_loader.preprocess import TranscriptPreprocessor
    from vector_store.vectorstore import VectorDB
    from data_loader.file_loader import Loader
except ImportError as e:
    print(f"❌ Lỗi import module: {e}")
    print("   Đảm bảo bạn đang chạy từ root project")
    sys.exit(1)


class DataPipeline:
    """Pipeline tự động: Crawl → Preprocess → Index"""
    
    def __init__(self):
        self.gpt_key = os.getenv("myAPIKey")
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        
        if not self.gpt_key:
            print("⚠️ Thiếu myAPIKey trong .env - cần cho embedding")
        if not self.gemini_key:
            print("⚠️ Thiếu GEMINI_API_KEY trong .env - sẽ bỏ qua sửa chính tả")
        
        # Paths
        self.root_data_dir = "data/"
        self.transcript_dir = "processed_transcripts/"
        self.metadata_dir = "metadata.json"
        self.output_dir = "chunks/"
    
    def step1_crawl_transcripts(self):
        """Bước 1: Crawl transcript từ YouTube"""
        print("\n" + "=" * 70)
        print("📥 BƯỚC 1: CRAWL TRANSCRIPTS TỪ YOUTUBE")
        print("=" * 70 + "\n")
        
        try:
            coordinator = ConfigBasedCoordinator()
            coordinator.process_all_enabled_playlists()
            print("\n✅ Hoàn tất crawl transcripts")
            return True
        except Exception as e:
            print(f"\n❌ Lỗi crawl: {e}")
            return False
    
    def step2_preprocess_transcripts(self, force_refetch: bool = False, playlist: str = None):
        """Bước 2: Preprocess transcript (validate + sửa lỗi)"""
        print("\n" + "=" * 70)
        print("🔧 BƯỚC 2: PREPROCESS TRANSCRIPTS")
        print("=" * 70 + "\n")
        
        try:
            preprocessor = TranscriptPreprocessor(gemini_api_key=self.gemini_key)
            
            if playlist:
                from pathlib import Path
                playlist_folder = Path("data") / playlist
                if not playlist_folder.exists():
                    print(f"❌ Không tìm thấy playlist: {playlist}")
                    return False
                preprocessor.process_playlist(playlist_folder, force_refetch)
            else:
                preprocessor.process_all_playlists(force_refetch)
            
            print("\n✅ Hoàn tất preprocess")
            return True
        except Exception as e:
            print(f"\n❌ Lỗi preprocess: {e}")
            return False
    
    def step3_index_to_vectordb(self):
        """Bước 3: Index vào Vector Database"""
        print("\n" + "=" * 70)
        print("🗄️ BƯỚC 3: INDEX VÀO VECTOR DATABASE")
        print("=" * 70 + "\n")
        
        try:
            if not self.gpt_key:
                print("❌ Thiếu myAPIKey - không thể tạo embedding")
                return False
            
            vector_db = VectorDB().db
            loader = Loader(
                open_api_key=self.gpt_key,
                vector_db=vector_db
            )
            
            print("📂 Đang load và chunk documents...")
            chunks = loader.load_dir(
                root_data_dir=self.root_data_dir,
                transcript_dir=self.transcript_dir,
                metadata_dir=self.metadata_dir,
                output_dir=self.output_dir,
                workers=2
            )
            
            print(f"📊 Đã tạo {len(chunks)} chunks")
            
            if chunks:
                print("💾 Đang index vào vector database...")
                vector_db.add_documents(chunks)
                print(f"✅ Đã index {len(chunks)} documents")
                return True
            else:
                print("⚠️ Không có chunks nào để index")
                return False
                
        except Exception as e:
            print(f"\n❌ Lỗi indexing: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_full_pipeline(
        self, 
        skip_crawl: bool = False,
        skip_preprocess: bool = False,
        skip_index: bool = False,
        force_refetch: bool = False,
        playlist: str = None
    ):
        """Chạy toàn bộ pipeline"""
        print("\n" + "🚀" * 35)
        print("🚀 BẮT ĐẦU PIPELINE: CRAWL → PREPROCESS → INDEX")
        print("🚀" * 35 + "\n")
        
        # Step 1: Crawl
        if not skip_crawl:
            success = self.step1_crawl_transcripts()
            if not success:
                print("\n⚠️ Crawl thất bại, tiếp tục với data hiện có...")
        else:
            print("\n⏭️ Bỏ qua bước crawl")
        
        # Step 2: Preprocess
        if not skip_preprocess:
            success = self.step2_preprocess_transcripts(force_refetch, playlist)
            if not success:
                print("\n❌ Preprocess thất bại, dừng pipeline")
                return False
        else:
            print("\n⏭️ Bỏ qua bước preprocess")
        
        # Step 3: Index
        if not skip_index:
            success = self.step3_index_to_vectordb()
            if not success:
                print("\n❌ Indexing thất bại")
                return False
        else:
            print("\n⏭️ Bỏ qua bước indexing")
        
        print("\n" + "🎉" * 35)
        print("🎉 HOÀN THÀNH TOÀN BỘ PIPELINE!")
        print("🎉" * 35 + "\n")
        return True


# =====================================================================
# CLI
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Data Pipeline: Crawl → Preprocess → Index",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  # Chạy full pipeline
  python pipeline.py
  
  # Bỏ qua crawl (nếu đã có transcript)
  python pipeline.py --skip-crawl
  
  # Bỏ qua preprocess
  python pipeline.py --skip-preprocess
  
  # Chỉ chạy indexing
  python pipeline.py --only-index
  
  # Chạy preprocess + index cho 1 playlist
  python pipeline.py --skip-crawl --playlist "cs431-cac-ki-thuat-hoc-sau-va-ung-dung"
  
  # Force refetch transcript lỗi
  python pipeline.py --skip-crawl --force-refetch
        """
    )
    
    parser.add_argument(
        "--skip-crawl",
        action="store_true",
        help="Bỏ qua bước crawl transcript"
    )
    
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Bỏ qua bước preprocess"
    )
    
    parser.add_argument(
        "--skip-index",
        action="store_true",
        help="Bỏ qua bước indexing"
    )
    
    parser.add_argument(
        "--only-index",
        action="store_true",
        help="Chỉ chạy indexing (bỏ qua crawl + preprocess)"
    )
    
    parser.add_argument(
        "--force-refetch",
        action="store_true",
        help="Force refetch transcript bị lỗi bằng Whisper"
    )
    
    parser.add_argument(
        "--playlist",
        type=str,
        help="Chỉ xử lý 1 playlist cụ thể (folder name)"
    )
    
    args = parser.parse_args()
    
    # Xử lý --only-index
    if args.only_index:
        args.skip_crawl = True
        args.skip_preprocess = True
    
    # Chạy pipeline
    pipeline = DataPipeline()
    success = pipeline.run_full_pipeline(
        skip_crawl=args.skip_crawl,
        skip_preprocess=args.skip_preprocess,
        skip_index=args.skip_index,
        force_refetch=args.force_refetch,
        playlist=args.playlist
    )
    
    sys.exit(0 if success else 1)