import os
import yaml
from .gpt import GPT

class Extract_Context(GPT):
    def __init__(
        self
    ) -> None:
        super().__init__()
        fp = os.path.join(os.path.dirname(__file__), 'prompt.yml')
        prompt = yaml.safe_load(open(fp, encoding='utf-8'))['image_to_text_prompt']
        self.system_message = prompt[0]['system_message']

    def _generate(
        self,
        context: str = '',
        img_url: str = None
    ) -> str:
        response = super()._generate(
            query=context,
            img_url=img_url
        )
        return response