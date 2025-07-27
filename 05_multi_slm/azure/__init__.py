import os
import yaml
from openai import AzureOpenAI

def get_auth():
    curr_dir = os.path.dirname(__file__)
    auth_path = os.path.join(curr_dir, 'auth.yml')
    auth = yaml.safe_load(open(auth_path, encoding='utf-8'))
    return auth

auth = get_auth()
endpoint = f"https://{auth['Azure_OpenAI']['name']}.cognitiveservices.azure.com/"
subscription_key = auth['Azure_OpenAI']['key']
api_version = "2024-12-01-preview"

e_endpoint = f"https://{auth['Azure_Embedding']['name']}.openai.azure.com/"
e_subscription_key = auth['Azure_Embedding']['key']
e_api_version = "2024-02-01"


client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

embedding_client = AzureOpenAI(
    api_version = e_api_version,
    azure_endpoint = e_endpoint,
    api_key = e_subscription_key,
)

def generate_embeddings(texts, model="text-embedding-ada-002", batch_size=16):
    """
    OpenAI 1.x SDK 기반 Azure OpenAI용 임베딩 함수
    - 단일 문자열 또는 문자열 리스트 처리
    - 자동 배치 처리
    """
    if isinstance(texts, str):
        texts = [texts]
        response = embedding_client.embeddings.create(input=texts, model=model)
        return response.data[0].embedding

    if not isinstance(texts, list):
        raise ValueError("Input must be a string or a list of strings.")

    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = embedding_client.embeddings.create(input=batch, model=model)
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

    return all_embeddings
