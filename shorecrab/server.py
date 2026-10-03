"""Serial local inference service. No arbitrary filesystem or URL image access."""
import argparse,base64,binascii,hmac,io,json,os,logging
from http.server import BaseHTTPRequestHandler,HTTPServer

def make_handler(runtime,api_key=None):
    class Handler(BaseHTTPRequestHandler):
        def send_json(self,status,body):
            data=json.dumps(body).encode();self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
        def do_GET(self):
            if self.path=='/health':self.send_json(200,{'status':'ready','model':'shorecrab-128m-a60'})
            else:self.send_json(404,{'error':'not found'})
        def do_POST(self):
            if self.path!='/v1/decisions':return self.send_json(404,{'error':'not found'})
            if api_key and not hmac.compare_digest(self.headers.get('Authorization',''),'Bearer '+api_key):return self.send_json(401,{'error':'unauthorized'})
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=24*1024**2:raise ValueError('request too large or missing Content-Length')
                body=json.loads(self.rfile.read(length))
                if set(body)!={'image','question','choices'}:raise ValueError('expected image, question, choices')
                value=body['image']
                if not isinstance(value,str) or not value.startswith(('data:image/png;base64,','data:image/jpeg;base64,','data:image/webp;base64,')):raise ValueError('base64 PNG, JPEG or WebP data URI required')
                raw=base64.b64decode(value.split(',',1)[1],validate=True)
                if len(raw)>16*1024**2:raise ValueError('image too large')
                from PIL import Image,ImageOps
                with Image.open(io.BytesIO(raw)) as im:
                    if im.width*im.height>64_000_000 or getattr(im,'n_frames',1)!=1:raise ValueError('image dimensions or animation unsupported')
                    im=ImageOps.exif_transpose(im)
                    if 'A' in im.getbands():
                        bg=Image.new('RGBA',im.size,(124,116,104,255));bg.alpha_composite(im.convert('RGBA'));image=bg.convert('RGB')
                    else:image=im.convert('RGB')
                    result=runtime.score(image,body['question'],body['choices'])
            except (ValueError,TypeError,KeyError,binascii.Error,OSError) as error:return self.send_json(400,{'error':str(error)})
            except Exception:
                logging.exception('Inference failed')
                return self.send_json(500,{'error':'inference failed; inspect server logs'})
            self.send_json(200,result)
        def log_message(self,fmt,*args):pass
    return Handler

def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',required=True);p.add_argument('--device',choices=['cpu','mps','cuda'],default='cpu');p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=8090);args=p.parse_args()
    key=os.getenv('CRAB_API_KEY')
    if args.host not in ['127.0.0.1','localhost','::1'] and not key:p.error('CRAB_API_KEY is required when binding outside localhost')
    import torch
    torch.set_num_threads(2)
    from .runtime import Runtime
    runtime=Runtime(args.bundle,args.device)
    with HTTPServer((args.host,args.port),make_handler(runtime,key)) as server:
        print(f'ShoreCrab ready at http://{args.host}:{args.port}',flush=True);server.serve_forever()

if __name__=='__main__':main()
