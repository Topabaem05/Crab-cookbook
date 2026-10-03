"""Read a game frame and print a proposed action; sends no game or OS input."""
import argparse,json,os
from shorecrab import CrabClient

CHOICES=['on the left','in the center','on the right','no enemy is visible']
ACTIONS=['turn_left','aim_at_center','turn_right','search']

def main():
    p=argparse.ArgumentParser();p.add_argument('image');args=p.parse_args()
    result=CrabClient(os.getenv('CRAB_BASE_URL','http://127.0.0.1:8090')).score(args.image,'Where is the nearest enemy on the screen?',CHOICES)
    index=result['answer_index'];action='hold' if index is None else ACTIONS[index]
    print(json.dumps({'proposed_action':action,'decision':result},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
