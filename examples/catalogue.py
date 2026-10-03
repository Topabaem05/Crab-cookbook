"""Select a supplied catalogue item from a camera frame."""
import argparse,json,os
from shorecrab import CrabClient

def main():
    p=argparse.ArgumentParser();p.add_argument('image');p.add_argument('--items',nargs='+',required=True);args=p.parse_args()
    result=CrabClient(os.getenv('CRAB_BASE_URL','http://127.0.0.1:8090')).score(args.image,'Which listed item is visible?',args.items)
    print(json.dumps({'catalogue_item':result['answer'],'decision':result},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
