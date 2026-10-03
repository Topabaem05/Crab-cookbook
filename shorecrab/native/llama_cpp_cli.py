"""Prepare inputs and invoke the native libllama A60 extension; inference stays in C++."""
import argparse,json,subprocess,tempfile
from pathlib import Path
from ..outputs import probability_view,select_answer

def main():
    p=argparse.ArgumentParser();p.add_argument('--executable',type=Path,required=True);p.add_argument('--model',type=Path,required=True)
    p.add_argument('--tokenizer',type=Path,required=True);p.add_argument('--image',type=Path,required=True);p.add_argument('--question',required=True)
    p.add_argument('--choices',nargs='+',required=True);p.add_argument('--output-mode',choices=['full','probabilities','select'],default='full');a=p.parse_args()
    from .inputs import prepare
    from .prepare_gguf import write_inputs
    from ..geometry import open_image,Tiling,Admission
    image=open_image(a.image,Tiling(locals='beyond_canvas'),Admission())
    data=prepare(image,a.question,a.choices,a.tokenizer)
    with tempfile.TemporaryDirectory(prefix='shorecrab-input-') as folder:
        inputs=Path(folder)/'input.gguf';output=Path(folder)/'result.json';write_inputs(data,inputs)
        subprocess.run([str(a.executable.resolve()),str(a.model.resolve()),str(inputs),str(output)],check=True)
        result=json.loads(output.read_text())
    if a.output_mode=='probabilities':result=probability_view(a.choices,result)
    elif a.output_mode=='select':result=select_answer(a.choices,result)
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
