"""Run A60 inside vLLM's native model worker (no inference HTTP service)."""
import argparse,json,os
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--image',type=Path,required=True);p.add_argument('--question',required=True);p.add_argument('--choices',nargs='+',required=True);a=p.parse_args()
    os.environ.setdefault('VLLM_PLUGINS','shorecrab_a60')
    os.environ.setdefault('VLLM_CPU_KVCACHE_SPACE','0');os.environ.setdefault('VLLM_CPU_OMP_THREADS_BIND','nobind')
    import torch
    from vllm import LLM
    from .inputs import prepare
    from ..geometry import open_image,Tiling,Admission
    torch.set_num_threads(2)
    image=open_image(a.image,Tiling(locals='beyond_canvas'),Admission())
    data=prepare(image,a.question,a.choices,a.model/'tokenizer')
    llm=LLM(model=str(a.model.resolve()),runner='pooling',dtype='float32',skip_tokenizer_init=True,enforce_eager=True,max_num_seqs=1,max_model_len=2,max_num_batched_tokens=2,enable_mm_embeds=True,limit_mm_per_prompt={'image':1},enable_prefix_caching=False,mm_processor_cache_gb=0)
    result=llm.encode({'prompt_token_ids':[1],'multi_modal_data':{'image':data}},pooling_task='plugin',use_tqdm=False)[0].outputs.data.tolist()
    k=int(result[17]);probs=result[:k];none=result[16];winner=max(range(k+1),key=(probs+[none]).__getitem__)
    print(json.dumps(dict(answer=a.choices[winner] if winner<k else None,probabilities=probs,none_probability=none,calibrated=False),ensure_ascii=False))
if __name__=='__main__':main()
