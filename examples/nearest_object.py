"""One image request, then display probabilities and the selected nearest object."""
import argparse
import json
import os
from shorecrab import CrabClient
from shorecrab.outputs import probability_view, select_answer

parser = argparse.ArgumentParser()
parser.add_argument("image")
parser.add_argument("--choices", nargs="+", default=["red car", "blue car", "traffic cone", "yellow barrier"])
args = parser.parse_args()
result = CrabClient(os.getenv("CRAB_BASE_URL", "http://127.0.0.1:8090")).score(args.image, "Which object is closest to the camera?", args.choices)
print(json.dumps(probability_view(args.choices, result), indent=2))
print(json.dumps(select_answer(args.choices, result), indent=2))
