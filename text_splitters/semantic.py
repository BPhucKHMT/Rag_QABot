from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from typing import List
import json
import os

class TranscriptChunker:
    """Chunk văn bản bằng SemanticChunker."""
    def __init__(self, open_api_key: str):
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            openai_api_key=open_api_key
        )
        self.splitter = SemanticChunker(
            embeddings=self.embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=85,
            min_chunk_size=300,
            add_start_index=True,
            buffer_size=2 
        )

    def __call__(self, documents: List[dict], output_dir: str) -> List[dict]:
        all_chunks = []
        for item in documents:
            full_text = item["full_text"]
            position_map = item["position_map"]
            playlist = item["playlist"]
            filename = item["filename"]
            title = item["title"]
            url = item["url"]


            chunks = self.splitter.create_documents(
                texts=[full_text],
                metadatas=[{
                    "playlist": playlist,
                    "video_url": url,
                    "filename": filename,
                    "title": title
                }]
            )
            for i, chunk in enumerate(chunks):
                start_index = chunk.metadata.pop("start_index")
                end_index = start_index + len(chunk.page_content)
                matched_ts = [
                    pos for pos in position_map
                    if not (pos["pos_end"] < start_index or pos["pos_start"] > end_index)
                ]
                if matched_ts:
                    chunk.metadata["start_timestamp"] = matched_ts[0]["start"]
                    chunk.metadata["end_timestamp"] = matched_ts[-1]["end"]
                else:
                    chunk.metadata["start_timestamp"] = None
                    chunk.metadata["end_timestamp"] = None
                chunk.metadata["chunk_id"] = i
            all_chunks.extend(chunks)

        output_path = os.path.join(output_dir, "semantic_chunks.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump([
                {"page_content": chunk.page_content, "metadata": chunk.metadata}
                for chunk in all_chunks
            ], f, ensure_ascii=False, indent=4)
        print(f"Saved {len(all_chunks)} chunks to {output_path}")
        return all_chunks
