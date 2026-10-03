"""python examples/ask_image.py assets/kitchen.png"""
import argparse,json,os
from shorecrab import CrabClient

def main():
    p=argparse.ArgumentParser();p.add_argument('image');p.add_argument('--question',default='What color is the mug?');p.add_argument('--choices',nargs='+',default=['white','red','blue','black']);p.add_argument('--url',default=os.getenv('CRAB_BASE_URL','http://127.0.0.1:8090'));args=p.parse_args()
    result=CrabClient(args.url).score(args.image,args.question,args.choices)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
