from typing import Union, List
import glob, os, re, multiprocessing
from tqdm import tqdm
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from dotenv import load_dotenv
from preprocess.preprocess import correct_spelling

load_dotenv()
gptkey = os.getenv("myAPIKey")


# ==========================================================
# Helper
# ==========================================================
def get_num_cpu():
    return multiprocessing.cpu_count()


# ==========================================================
# TranscriptParser: chỉ lo parse + map timestamp
# ==========================================================
class TranscriptParser:
    def __init__(self, embedding):
        self.splitter = SemanticChunker(
            embeddings=embedding,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=90,
            min_chunk_size=500,
            add_start_index=True,
            buffer_size=1
        )

    def parse_transcript(self, file_path: str):
        """Đọc file transcript, tách từng dòng có start-end-text"""
        timestamps = []
        full_text = ""
        position_map = []

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or "[âm nhạc]" in line.lower():
                    continue
                filename = os.path.basename(file_path).replace(".txt", "")
                match = re.match(r"(\d+:\d+:\d+)\s*-\s*(\d+:\d+:\d+),\s*(.+)", line)
                if match:
                    start, end, text = match.groups()
                    pos = len(full_text)
                    full_text += text + " "
                    position_map.append({
                        "start": start,
                        "end": end,
                        "text": text,
                        "pos_start": pos,
                        "pos_end": len(full_text)
                    })

        return full_text.strip(), position_map, filename

    def chunk(self, file_path: str, video_url: str = None):
        """Semantic chunk + mapping timestamp"""
        text, position_map, filename = self.parse_transcript(file_path)
        text = correct_spelling(text)

        chunks = self.splitter.create_documents(
            texts=[text],
            metadatas=[{
                "video_url": video_url,
                "filename": filename,
                "source_file": file_path
            }]
        )

        # map timestamp
        for i, chunk in enumerate(chunks):
            start_idx = chunk.metadata.get("start_index", 0)
            end_idx = start_idx + len(chunk.page_content)
            matched_ts = [
                ts for ts in position_map
                if not (ts["pos_end"] < start_idx or ts["pos_start"] > end_idx)
            ]
            if matched_ts:
                chunk.metadata["start_timestamp"] = matched_ts[0]["start"]
                chunk.metadata["end_timestamp"] = matched_ts[-1]["end"]
            else:
                chunk.metadata["start_timestamp"] = None
                chunk.metadata["end_timestamp"] = None
            chunk.metadata["chunk_id"] = i

        return chunks


# ==========================================================
# BaseLoader
# ==========================================================
class BaseLoader:
    def __init__(self):
        self.num_processes = get_num_cpu()

    def __call__(self, files: List[str], **kwargs):
        pass


# ==========================================================
# TranscriptLoader (giữ nguyên logic cũ + tích hợp parser)
# ==========================================================
class TranscriptLoader(BaseLoader):
    def __init__(self):
        super().__init__()
        self.embedding = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=gptkey)
        self.parser = TranscriptParser(self.embedding)

    def load_transcript_plain(self, file_path: str):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        corrected = correct_spelling(text)
        return Document(page_content=corrected, metadata={"filename": os.path.basename(file_path)})

    def __call__(self, files: List[str], **kwargs):
        """Tự động chọn cách load"""
        all_docs = []
        for file_path in tqdm(files, desc="Loading transcripts"):
            with open(file_path, "r", encoding="utf-8") as f:
                head = f.readline()
            # nếu dòng đầu có timestamp -> parse theo format có timestamp
            if re.match(r"\d+:\d+:\d+\s*-\s*\d+:\d+:\d+", head):
                docs = self.parser.chunk(file_path, kwargs.get("video_url"))
                all_docs.extend(docs)
            else:
                all_docs.append(self.load_transcript_plain(file_path))
        return all_docs


# ==========================================================
# Loader (wrapper cuối)
# ==========================================================
class Loader:
    def __init__(self):
        self.loader = TranscriptLoader()

    def load_dir(self, dir_path: str, workers: int = 1, video_url: str = None):
        txt_files = glob.glob(f"{dir_path}/*.txt")
        assert len(txt_files) > 0, "Không tìm thấy file txt trong thư mục"
        return self.loader(txt_files, workers=workers, video_url=video_url)


# ==========================================================
# Test
# ==========================================================
if __name__ == "__main__":
    loader = Loader()
    docs = loader.load_dir("./data", video_url="https://youtube.com/v123")
    print(docs[0].metadata)
    print(docs[0].page_content[:300])
