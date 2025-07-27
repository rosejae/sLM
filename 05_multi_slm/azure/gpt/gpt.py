from .. import client
from openai._exceptions import RateLimitError, APITimeoutError, OpenAIError

class GPT():
    def __init__(self) -> None:
        self.engine='gpt-4o'
        self.temperature=0.7
        self.few_shot_examples=None
        
    def _generate(
        self,
        query: str = None,  
        img_url: str = None,       
    ) -> str:
        messages = [{'role':'system',
                     'content': f'{self.system_message}'
                     }]
        if self.few_shot_examples is not None:
            for example in self.few_shot_examples:
                messages.append({'role':'user', 'content':f"{example['user_message']}"})
                messages.append({'role':'assistant', 'content':f"{example['assistant_message']}"})
        if img_url is None:    
            messages.append({'role':'user', 'content':f"{query}"})
        else:
            messages.append(
            {
            "role":"user", 
            "content": 
                [
                {
                    "type": "text",
                    "text" : query,
                    },
                {
                    "type": "image_url",
                    "image_url":{
                        "url": img_url,
                        }
                    }
                ]
            }
            )             
        try:
            response = client.chat.completions.create(
                model=self.engine,
                messages=messages,
                temperature=self.temperature,
            ).choices[0].message.content
            return response
        except RateLimitError:
            print("OpenAI API RateLimitError occurred!")
            return '[Error]'
        except APITimeoutError:
            print("OpenAI API Timeout occurred!")
            return '[Error]'
        except OpenAIError as e:
            print(f"기타 OpenAI API 오류: {e}")
            return '[Error]'