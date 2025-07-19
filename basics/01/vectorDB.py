#
# chroma - langchain not used 
#

import pandas as pd
from tqdm import tqdm
from datasets import load_dataset

import chromadb
from sentence_transformers import SentenceTransformer

df = load_dataset("daekeun-ml/naver-news-summarization-ko", split='train[:20]').to_pandas()

#chroma_client = chromadb.PersistentClient()
#chroma_client.delete_collection(name="my_collection")

chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="my_collection")
model = SentenceTransformer('snunlp/KR-SBERT-V40K-KlueNLI-augSTS')

ids = []
metadatas = []
embeddings = []

for row in tqdm(df.iterrows()):
    index = row[0]
    query = row[1].title
    answer = row[1].summary

    metadata = {
        "query": query,
        "answer": answer
    }

    embedding = model.encode(query, normalize_embeddings=True)

    ids.append(str(index))
    metadatas.append(metadata)
    embeddings.append(embedding)

chunk_size = 1024  # 한 번에 처리할 chunk 크기 설정
total_chunks = len(embeddings) // chunk_size + 1  # 전체 데이터를 chunk 단위로 나눈 횟수
embeddings = [e.tolist() for e in tqdm(embeddings)]

for chunk_idx in tqdm(range(total_chunks)):
    start_idx = chunk_idx * chunk_size
    end_idx = (chunk_idx + 1) * chunk_size

    # chunk 단위로 데이터 자르기
    chunk_embeddings = embeddings[start_idx:end_idx]
    chunk_ids = ids[start_idx:end_idx]
    chunk_metadatas = metadatas[start_idx:end_idx]

    # chunk를 answers에 추가
    collection.add(embeddings=chunk_embeddings, ids=chunk_ids, metadatas=chunk_metadatas)

result = collection.query(
    query_embeddings=model.encode("미국 뉴욕증시 52년만에", normalize_embeddings=True).tolist(),
    n_results=3
    )

print(result)

#
# chroma with langchain
#

import os 

from langchain_chroma import Chroma
# from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import CSVLoader
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_text_splitters import CharacterTextSplitter

if not os.path.exists("naver_news.csv"):
    dataset = load_dataset("daekeun-ml/naver-news-summarization-ko", split='train[:20]')
    dataset.to_csv("naver_news.csv", index=False)

loader = CSVLoader("naver_news.csv", encoding='utf-8')
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma.from_documents(docs, embedding_function)

query = "에디슨이노 이승훈"
docs = db.similarity_search(query)

print(docs[0].page_content)

#
# FAISS
#

import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

df = pd.read_csv("test.csv")
texts = df.to_numpy()

titles = [item[3] for item in texts]
embedder = SentenceTransformer('Huffon/sentence-klue-roberta-base')
vectors = embedder.encode(titles, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True)

index = faiss.IndexFlatIP(vectors.shape[1])
index.add(vectors) 

top_k = 4
query = "세입자 구해오라는 집주인"
query_embedding = embedder.encode(query, normalize_embeddings=True, convert_to_numpy=True)
distances, indices = index.search(np.expand_dims(query_embedding, axis=0), k=top_k)

temp = df.iloc[indices[0]]
temp['distance'] = distances[0]
temp[['title', 'document', 'link', 'summary']]