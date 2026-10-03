"""Small HTTP client; importing it does not load model dependencies."""
import base64,json,mimetypes,os
from pathlib import Path
from urllib.request import Request,urlopen

class CrabClient:
    def __init__(self,base_url='http://127.0.0.1:8090',api_key=None,timeout=120):
        self.base_url=base_url.rstrip('/');self.api_key=api_key or os.getenv('CRAB_API_KEY');self.timeout=timeout
    def score(self,image,question,choices):
        path=Path(image);mime=mimetypes.guess_type(path.name)[0]
        if mime not in ['image/png','image/jpeg','image/webp']:raise ValueError('PNG, JPEG or WebP required')
        if path.stat().st_size>16*1024**2:raise ValueError('image exceeds 16 MiB')
        payload=dict(image='data:'+mime+';base64,'+base64.b64encode(path.read_bytes()).decode(),question=question,choices=choices)
        headers={'Content-Type':'application/json'}
        if self.api_key:headers['Authorization']='Bearer '+self.api_key
        request=Request(self.base_url+'/v1/decisions',data=json.dumps(payload).encode(),headers=headers,method='POST')
        with urlopen(request,timeout=self.timeout) as response:return json.load(response)
    def choose(self,image,question,choices):
        """Select the largest joint probability, preserving a winning None outcome."""
        from .outputs import select_answer
        return select_answer(choices,self.score(image,question,choices))
