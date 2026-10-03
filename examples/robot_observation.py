"""Read a robot camera frame and print an object selection for inspection."""
import argparse,json,os
from shorecrab import CrabClient

def main():
    p=argparse.ArgumentParser();p.add_argument('image');p.add_argument('--question',default='Which object is blue?');p.add_argument('--objects',nargs='+',default=['a mug','a cube','a ball','a bottle']);args=p.parse_args()
    result=CrabClient(os.getenv('CRAB_BASE_URL','http://127.0.0.1:8090')).score(args.image,args.question,args.objects)
    print(json.dumps({'selected_object':result['answer'],'decision':result},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
