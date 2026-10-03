"""Compare relative object distance; this endpoint does not return a depth map."""
import argparse,json,os
from shorecrab import CrabClient

def main():
    p=argparse.ArgumentParser();p.add_argument('image');p.add_argument('first');p.add_argument('second');args=p.parse_args()
    result=CrabClient(os.getenv('CRAB_BASE_URL','http://127.0.0.1:8090')).score(args.image,'Which object is closer to the camera?',[args.first,args.second])
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
