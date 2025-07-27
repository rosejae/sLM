import os
import yaml
import json
import re

import gradio as gr

from azure.gpt.orch import Orch
from azure.google_search.news import News
from azure.google_search.web import Web
from azure.cognitive_search.ai_search import AISearch
from huggingface.inference_Llama3 import Inference

orch = Orch()
news = News()
web = Web()
ai_search = AISearch()
infer = Inference()

# 링크 및 이미지 HTML 만드는 클래스
class HTMLGenerator:  
    def __init__(self):  
        pass  
    
    @classmethod 
    def href_link(cls, name, url):  
        return f"<a href='{url}' style='color: blue'>{name}</a>"  
    
    @classmethod 
    def image_html(cls, image_path, caption):  
        return f"<a href='file://{image_path}' target='_blank'><img src='/file={image_path}' alt='Image One'></a><div>{caption}</div>"  

class Module:  
    fp = os.path.join(os.path.dirname(__file__), 'huggingface/prompt.yml')
    prompt_llama = yaml.safe_load(open(fp, encoding='utf-8'))

    def __init__(self):  
        pass

    # orchestration module
    @classmethod 
    def _orch_generate(
        cls, 
        message: dict, 
        history: list
        ) -> tuple:

        user_text = message['text']
        if len(message['files']) > 0:
            file_name = message['files'][0].split('/')[-1] 
        else:
            file_name = ''
        
        orch_prompt = ''

        for h in history[-2:]:
            #문서와 질문이 같이 들어 올 경우, 따로 처리를 해준다.
            if h[1] == None:
                history[-1][0] += "\n문서 이름 :"
                history[-1][0] += h[0][0].split('/')[-1]
                history.remove(h)
                continue
            orch_prompt += ('user : ' + h[0] + '\n')
            orch_prompt += ('assistant : ' + h[1] + '\n')
            #orch_prompt에서 링크 제거
            orch_prompt = re.sub(r'<a .*?>.*?</a>', '', orch_prompt)  
        orch_prompt += ('user : ' + user_text)

        #문서가 같이 들어온 경우 따로 처리
        if file_name != '':
            orch_prompt += f"\n문서 이름 :{file_name}"

        print(f"orch prompt: {orch_prompt}")
        print("\n=============================================\n")

        response = json.loads(orch._generate(orch_prompt))
        print(f"orch response: {response}")
        print("\n=============================================\n")

        agent = response['agent']
        keyword = response['keyword']
        print(f"agent: {agent}\nkeyword: {keyword}")
        print("\n=============================================\n")

        return agent, keyword

    # generator response module
    @classmethod 
    def _response_generate(
        cls,
        message: str,
        search_results: list,
        agent: str,
        keyword: str,
        args: dict
    ) -> str:

        user_text = message['text']

        if "News agent" == agent:
            system_prompt = cls.prompt_llama['news_prompt'][0]['system_massage']
            fewshot_prompt = cls.prompt_llama['news_prompt'][1:]
        elif "Web agent" == agent:
            system_prompt = cls.prompt_llama['web_prompt'][0]['system_massage']
            fewshot_prompt = cls.prompt_llama['web_prompt'][1:]
        elif "Document" in agent:
            system_prompt = cls.prompt_llama['document_prompt'][0]['system_massage']
            fewshot_prompt = cls.prompt_llama['document_prompt'][1:]

        for token in infer._generate(
            system_prompt=system_prompt,
            fewshot_prompt=fewshot_prompt,
            context_prompt=search_results,
            question_prompt=user_text,
            keyword=keyword,
            args=args
        ):
            yield token

# agent modules (tools)
class Retrieval:  
    def __init__(self):  
        pass  

    #AI Search
    @classmethod
    def _aisearch(
        cls,
        keyword: str, 
        file_name: str,
        count: int=3
    ) -> list:
        retrieval_results = ai_search.search(
            query=keyword,
            file_name=file_name,
            topk=count
        )
        text = ''
        for i, d in enumerate(retrieval_results):
            text += f"\n검색문서 {i+1}\ncontent: {d['context']}"

        return text

    #빙 웹검색 API
    @classmethod
    def _newssearch(
        cls,
        keyword: str,
        count: str=3
    ) -> str:
        search_results = news.search(
            query=keyword,
            count=count
        )

        text = ''
        urls = []
        titles = []
        for i, d in enumerate(search_results):
            text += f"\n뉴스 {i+1}\ntitle: {d['title']}"
            text += f"\nsnippet:{d['snippet']}"
            urls.append(d['url'])
            titles.append(d['title'])

        return text, urls, titles

    #빙 뉴스검색 API
    @classmethod
    def _websearch(
        cls,
        keyword: str,
        count: str=3
    ) -> str:

        search_results = web.search(
            query=keyword,
            count=count
        )

        text = ''
        urls = []
        titles = []
        for i, d in enumerate(search_results):
            text += f"\n검색결과 {i+1}\ntitle: {d['title']}"
            text += f"\nsnippet:{d['snippet']}"
            urls.append(d['url'])
            titles.append(d['title'])
        return text, urls, titles

def run(message, history):
    print(f"message : {message}")
    print(f"history : {history}")
    print("\n=============================================\n")
    
    agent, keyword = Module._orch_generate(message, history)

    if "Document" in agent:
        file_name = agent.split('Document:')[1].strip()
        print(f"file name : {file_name}")
        search_results = Retrieval._aisearch(
            keyword=keyword,
            file_name=file_name,
            count=max_search_document.value
        )

    elif agent == "Web agent":
        search_results, urls, titles = Retrieval._websearch(
            keyword=keyword,
            count=max_search_document.value
        )
    
    elif agent == "News agent":
        search_results, urls, titles = Retrieval._newssearch(
            keyword=keyword,
            count=max_search_document.value
        )

    print(f"search_results : {search_results}")
    print("\n=============================================\n")
    
    args = {
        "temperature" : temperature.value,
        "top_p" : top_p.value,
        "top_k" : top_k.value,
        "max_new_tokens" : max_new_tokens.value
    }
    
    response = ''
    for token in Module._response_generate(
        message=message,
        search_results=search_results,
        agent=agent,
        keyword=keyword,
        args=args
    ):
        response += token
        yield response
    
    if (agent == "Web agent") or (agent == "News agent"):
        for url, title in zip(urls, titles):
            response += "\n"
            response += HTMLGenerator.href_link(
                name=title[:20],
                url=url
            )
            yield response

    # 챗봇에 이미지 띄우기 예시
    # if "Document" in agent:
    #     response += "\n"
    #     image_path = os.path.join(os.path.dirname(__file__),"azure/cognitive_search/냉장고_메뉴얼2_image/냉장고_메뉴얼2_page_14.png")
    #     response += HTMLGenerator.image_html(image_path, "참고자료1")
    #     yield response

def update_value(val, component_name):
    component = globals()[component_name]
    component.value = val
    return f'Value is set to {val}'

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown('<h1>나만의 업무 AI 비서</h1>')
            chat1 = gr.ChatInterface(
                fn=run,
                multimodal=True,
                fill_height=True,
                theme='soft',
                # clear_btn='Clear',
                examples=[
                    {'text': '최신 AI 기술에 대한 뉴스 찾아줄래?'},
                    {'text': '이 문서 요약해줘'}
                ]                
            )
        with gr.Column(scale=1):
            with gr.Row():
                gr.Markdown("<h3 style='margin-top: 10px'>Model Parameters</h3>")
                
            model = gr.Dropdown(
                value='Llama3(Ours)',
                choices=['Llama3(Ours)', 'GPT-3.5', 'GPT-4'],
                label='Model',
                info='Choose a model',
                interactive=False
            )
            
            temperature = gr.Slider(
                value=0.7,
                maximum=1.0,
                label='Temperature',
                interactive=True,
                info='''The value used to modulate the next token probabilities.'''
            )
            
            temperature.change(
                fn=update_value,
                inputs=[temperature, gr.State('temperature')]
            )

            top_p = gr.Slider(  
                value=0.7,  
                maximum=1.0,  
                label="Top-p",  
                interactive=True,  
                info="""If set to float < 1, only the smallest set of most probable tokens   
                with probabilities that add up to top_p or higher are kept for generation."""  
            )

            top_p.change(  
                fn=update_value,  
                inputs=[top_p, gr.State('top_p')]
            ) 

            top_k = gr.Slider(  
                minimum=1,  
                value=50,  
                maximum=100,  
                step=1,  
                label="Top-k",  
                interactive=True,  
                info="""The number of highest probability vocabulary tokens to keep for top-k-filtering."""  
            )

            top_k.change(  
                fn=update_value,  
                inputs=[top_k, gr.State('top_k')]
            ) 

            max_new_tokens = gr.Slider(  
                minimum=50,  
                value=512,  
                maximum=2000,  
                step=1,  
                label="max_new_tokens",  
                interactive=True,  
                info="""This parameter controls the maximum number of new tokens   
                to be generated in each step of the model's output.  
                A lower value will result in shorter responses from the model."""  
            )

            max_new_tokens.change(  
                fn=update_value,  
                inputs=[max_new_tokens, gr.State('max_new_tokens')]
            )

            max_search_document = gr.Slider(  
                minimum=1,  
                value=3,  
                maximum=5,  
                step=1,  
                label="max_search_document",  
                interactive=True,  
                info="""This parameter determines the max number of documents to be 
                considered during the search process when the model is looking 
                for information to generate the output. Higher values may result 
                in more diverse but possibly less focused responses."""
            )

            max_search_document.change(  
                fn=update_value,  
                inputs=[max_search_document, gr.State('max_search_document')]
            ) 
            
if __name__ == '__main__':
    demo.queue()
    demo.launch(share=True, allowed_paths=['.'])            
            
            
            
            