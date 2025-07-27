import re
import requests

from .. import get_auth

class News():
    def __init__(self) -> None:
        auth = get_auth()
        self.endpoint = 'https://www.googleapis.com/customsearch/v1?'
        self.key = auth['Google_Search']['key']
        self.engine = auth['Google_Search']['news_engine']
        self.params = {
            'key': self.key,
            'cx': self.engine,
            'lr': 'lang_ko',
            'safe': 'active',
            'sort': 'date',
        }
        
    def search(
        self,
        query: str,
        num: int=3,
    ) -> list:
        for keyword in ['오늘', '주요', '뉴스']:
            query = query.replace(keyword, "").strip()
            
        if not query:
            query = '뉴스'
        
        # url = fr'{self.endpoint}key={self.key}&cx={self.engine}'
        url = self.endpoint
        self.params['q'] = f'{query}'
        self.params['num'] = num
        
        search_result = requests.get(
            url, 
            params=self.params,
            ).json()
        
        documents = []
        for item in search_result.get("items", []):
            title = item.get('title', 'no title')
            url = item.get('link', 'no url')
            snippet = re.sub(r"\[.*?\]", "", item.get('snippet', 'no content'))
            documents.append({
                'title': title,
                'snippet': snippet,
                'url': url,
            })
        return documents
            

