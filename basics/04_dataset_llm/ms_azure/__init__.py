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

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)