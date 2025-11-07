from typing import Union,List, Literal
import glob
from tqdm import tqdm
import multiprocessing


from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
import re
import torch

from dotenv import load_dotenv
import os

from preprocess.preprocess import correct_spelling # sửa chính tả
# Load environment variables from .env in project root
load_dotenv()

# Read and normalize API key (strip optional quotes and surrounding whitespace)
gptkey = os.getenv('myAPIKey')

## load txt file
def load_transcript(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        transcript = file.read()
    # Sửa chính tả cho toàn bộ transcript
    corrected_transcript = correct_spelling(transcript)
    return corrected_transcript

def get_num_cpu():
    return multiprocessing.cpu_count()

class BaseLoader():
    def __init__(self) -> None:
        self.num_processes = get_num_cpu()
    def __call__(self, files: List[str], **kwargs) -> None:
        pass

class TranscriptLoader(BaseLoader):
    def __init__(self) -> None:
        super().__init__()
    def __call__(self, files: List[str], **kwargs) -> List[str]:
        num_processes = min(self.num_processes, kwargs["workers"])

        with multiprocessing.Pool(processes=num_processes) as pool:
            doc_loaded =[]
            total_files = len(files)
            with tqdm(total=total_files, desc="Loading transcripts") as pbar:
                for result in pool.imap_unordered(load_transcript, files):
                    doc_loaded.append(result)
                    pbar.update(1)
        return doc_loaded

class TextSplitter():
    def __init__(self, embedding_model = OpenAIEmbeddings( model="text-embedding-3-large",
            openai_api_key=gptkey), minChunkSize: int =500, threshold: float= 0.9, **kwargs) -> None:
        self.embedding = embedding_model
        self.splitter = SemanticChunker(
            embeddings=self.embedding,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=threshold,
            min_chunk_size= minChunkSize,
            add_start_index=True,
            buffer_size=1
        )
    def __call__(self,documents):
        return self.splitter.split_documents(documents)

class Loader:
    def __init__(self, file_type: str="txt", split_kwargs: dict={
        "minChunkSize": 500,
        "threshold": 0.9
    })-> None:
        assert file_type in ["txt"], "Chỉ hỗ trợ file txt"
        self.file_type = file_type
        if file_type == "txt":
            self.loader = TranscriptLoader()
        else:
            raise ValueError("file_type phải là txt")
        self.text_splitter = TextSplitter(**split_kwargs)

    def load(self, txt_files: Union[str, List[str]], workers: int =1):
        if isinstance(txt_files, str):
            txt_files = [txt_files]# chuyển thành list nếu là chuỗi đơn
        documents = self.loader(txt_files, workers=workers)
        doc_split = self.text_splitter(documents)
        return doc_split
    
    def load_dir(self, dir_path: str, workers: int =1):
        if self.file_type == "txt":
            txt_files = glob.glob(f"{dir_path}/*.txt")
            assert len(txt_files) >0, "Không tìm thấy file txt trong thư mục"
        else:
            raise ValueError("file_type phải là txt")
        return self.load(txt_files, workers=workers)

if __name__ == "__main__":
    loader = Loader()
    files = glob.glob("./data/*.txt")
    Load = TranscriptLoader()
    document = Load(files, workers=1)
