# Hệ thống hỏi đáp môn học CS315

## Cấu trúc thư mục
``` bash
├── data
├── database
│   ├── notebook_baseline
│   │   ├── pipeline.ipynb: pipeline toàn bộ chương trình
│   │   └── test.py
│   ├── preprocess
│   │   ├── preprocess_transcript.ipynb : sửa lỗi chính tả và format định dạng file txt transcript
│   │   └── preprocess.py: sửa lỗi chính tả
│   └── rag
│       ├── file_loader.py : load file txt transcript để chunk
│       ├── llm_model.py: define model
│       ├── main.py: tạo rag chain (chính)
│       ├── offline_rag.py: 
│       ├── reranking.py: rerank
│       └── vectorstore.py: load db
├── .env
├── .gitignore
├── README.md
└── requirements.txt
```
## Cha


