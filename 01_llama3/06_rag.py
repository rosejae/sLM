import torch
from torch import bfloat16
import transformers

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    AutoConfig,
)

#
# config
#

# quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16,
)

# model, tokenizer load
model_id = 'MLP-KTLim/llama-3-Korean-Bllossom-8B'
model_config = AutoConfig.from_pretrained(
    model_id,
    trust_remote_code=True,
    max_new_tokens=1024
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    quantization_config=bnb_config,
    device_map={"":0},
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

#
# inference
#

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map={"":0},
)

pipeline.model.eval()

prompt = "You are a helpful AI assistant. Please answer the user's questions kindly. 당신은 유능한 AI 어시스턴트 입니다. 사용자의 질문에 대해 친절하게 답변해주세요."
question = "서울의 유명한 관광 코스를 만들어줄래?"

messages = [
            {"role": "system", "content": f"{prompt}"},
            {"role": "user", "content": f"{question}"}
            ]

chat_prompt = pipeline.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
)

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = pipeline(
    chat_prompt,
    max_new_tokens=2048,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9
)

#
# dataset load (history.csv) 
#

import pandas as pd
from langchain_community.document_loaders import DataFrameLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

from langchain_chroma import Chroma
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma

df = pd.read_csv("/content/drive/MyDrive/part4/history.csv", encoding='utf-8-sig')

df = df[['extracted_text', 'translate_text']]
df = df[:100]
print(df.head(10))
loader = DataFrameLoader(df, page_content_column="extracted_text")
df = loader.load()

# text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
all_splits = text_splitter.split_documents(df)

# embedding model
model_name = "jhgan/ko-sroberta-multitask"
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = SentenceTransformerEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# vector database
vectordb = Chroma.from_documents(documents=all_splits, embedding=embeddings)
vectordb.get(include=["metadatas", "documents", "embeddings"], limit=10, offset=1)

# vectordb test
query = " Please translate the following sentence. Please construct the translation by referring to metadat. \n\"new\" British history and Atlantic history in the early 1970"
docs = vectordb.similarity_search(query)
print(docs[0].page_content)
print(docs[0].metadata)

# vectordb + llm -> retrievalQA
retriever = vectordb.as_retriever()
llm = HuggingFacePipeline(pipeline=pipeline)
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    verbose=True,
    return_source_documents=True,
)

# retrievalQA test
query = "Please translate the following sentences into Korean. Please construct the translation by referring to metadat. \n\"new\" British history and Atlantic history in the early 1970"

response = qa(query)
print(response)
print(response['source_documents'])


#
# dataset load (train_0.csv) - 고어 한국어 번역
#

df = pd.read_csv("/content/drive/MyDrive/part4/train_0.csv", encoding='utf-8-sig')
df = df[['original','modern translation']]
df = df[:100]
print(df.head(10))
loader = DataFrameLoader(df, page_content_column="modern translation")
df = loader.load()

# text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
all_splits = text_splitter.split_documents(df)

# embedding model
model_name = "jhgan/ko-sroberta-multitask"
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = SentenceTransformerEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)


vectordb = Chroma.from_documents(documents=all_splits, embedding=embeddings)
vectordb.get(include=["metadatas", "documents", "embeddings"], limit=10, offset=1)

query = "전하 그 덕망을 승히 여기사 벼슬을 돋우어 이조판서로 좌의정을 하게 하시니, 승상이 국은을 감동하여 갈충보국하니 사방에 일이 업고 도적이 없으매 시화연풍하여 나라가 태평하더라"
docs = vectordb.similarity_search(query)

print(docs[0].page_content)
print(docs[0].metadata)

retriever = vectordb.as_retriever()
llm = HuggingFacePipeline(pipeline=pipeline)
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    verbose=True,
    return_source_documents=True,
)

query = "전하 그 덕망을 승히 여기사 벼슬을 돋우어 이조판서로 좌의정을 하게 하시니, 승상이 국은을 감동하여 갈충보국하니 사방에 일이 업고 도적이 없으매 시화연풍하여 나라가 태평하더라"

response = qa(query)
print(response['source_documents'])
print(response['source_documents'][0])