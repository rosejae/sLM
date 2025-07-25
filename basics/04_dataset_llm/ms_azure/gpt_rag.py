import os
import yaml
from .gpt import GPT

class RAG(GPT):
    def __init__(
        self
    ) -> None:
        super().__init__()
        fp = os.path.join(os.path.dirname(__file__), 'prompt.yml')
        prompt = yaml.safe_load(open(fp, encoding='utf-8'))['rag_generate_prompt']
        self.system_message = prompt[0]['system_message']
        self.few_shot_examples = prompt[1:]

    def _generate(
        self,
        context: str = '',
        img_url: str = None
    ) -> str:
        response = super()._generate(query=f"context: {context}")
        return response