"""Weight-free image/token preprocessing for the C++ A60 runner."""
import argparse
from pathlib import Path
import numpy as np
from gguf import GGUFWriter
from ..geometry import open_image,Tiling,Admission
from .inputs import prepare

def write_inputs(data,path):
    writer=GGUFWriter(str(path),'shorecrab-input')
    for name,tensor in data.items():
        array=tensor.detach().cpu().numpy()
        if name.endswith('_ids') or name.endswith('_mask'):array=array.astype(np.int32)
        writer.add_tensor(name,array)
    writer.write_header_to_file();writer.write_kv_data_to_file();writer.write_tensors_to_file();writer.close()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--image',type=Path,required=True);p.add_argument('--tokenizer',type=Path,required=True);p.add_argument('--question',required=True);p.add_argument('--choices',nargs='+',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error('output exists')
    image=open_image(a.image,Tiling(locals='beyond_canvas'),Admission())
    write_inputs(prepare(image,a.question,a.choices,a.tokenizer),a.output)
