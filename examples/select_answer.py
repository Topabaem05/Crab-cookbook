"""Select the largest candidate-or-None probability from an existing local server."""
import argparse,json,os
from shorecrab import CrabClient

def main():
    p=argparse.ArgumentParser();p.add_argument('image');p.add_argument('--question',default='What color is the mug?')
    p.add_argument('--choices',nargs='+',default=['white','red','blue','black']);a=p.parse_args()
    client=CrabClient(os.getenv('CRAB_BASE_URL','http://127.0.0.1:8090'))
    print(json.dumps(client.choose(a.image,a.question,a.choices),indent=2))

if __name__=='__main__':main()
