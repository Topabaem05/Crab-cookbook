import base64,io,json,threading
from http.server import HTTPServer
from urllib.request import Request,urlopen
from urllib.error import HTTPError
import pytest
from PIL import Image
from shorecrab.server import make_handler
from shorecrab.client import CrabClient

class FixtureRuntime:
    """Protocol fixture, never a model-accuracy test."""
    def __init__(self):self.calls=[]
    def score(self,image,question,choices):
        self.calls.append((image.copy(),question,choices))
        return dict(answer=None,answer_index=None,probabilities=[.2,.3],none_probability=.5,calibrated=False)

@pytest.fixture
def endpoint():
    runtime=FixtureRuntime();server=HTTPServer(('127.0.0.1',0),make_handler(runtime,'test-only-key'))
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    yield f'http://127.0.0.1:{server.server_port}',runtime
    server.shutdown();server.server_close();thread.join()

def post(url,body,key='test-only-key'):
    request=Request(url+'/v1/decisions',data=json.dumps(body).encode(),headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'},method='POST')
    with urlopen(request) as response:return json.load(response)

def payload():
    buf=io.BytesIO();Image.new('RGB',(13,9),'red').save(buf,format='PNG')
    return dict(image='data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode(),question='Which colour?',choices=['blue','red'])

def test_roundtrip_preserves_pixels_candidates_and_none(endpoint,tmp_path):
    url,runtime=endpoint;p=tmp_path/'sample.png';Image.new('RGB',(13,9),'red').save(p)
    out=CrabClient(url,api_key='test-only-key').score(p,'Which colour?',['blue','red'])
    assert out['answer'] is None and out['probabilities']==[.2,.3]
    image,question,choices=runtime.calls[0]
    assert image.size==(13,9) and image.getpixel((0,0))==(255,0,0)
    assert choices==['blue','red'] and question=='Which colour?'

def test_bad_key_never_calls_model(endpoint):
    url,runtime=endpoint
    with pytest.raises(HTTPError) as error:post(url,payload(),key='wrong')
    assert error.value.code==401 and not runtime.calls

@pytest.mark.parametrize('image',['file:///etc/passwd','https://example.com/image.png','data:image/png;base64,%%%%'])
def test_invalid_image_sources_rejected_before_model(endpoint,image):
    url,runtime=endpoint;body=payload();body['image']=image
    with pytest.raises(HTTPError) as error:post(url,body)
    assert error.value.code==400 and not runtime.calls

def test_unknown_fields_not_ignored(endpoint):
    url,runtime=endpoint;body=payload();body['hidden_instruction']='unused'
    with pytest.raises(HTTPError) as error:post(url,body)
    assert error.value.code==400 and not runtime.calls
