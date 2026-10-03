"""Load an authorized A60 inference bundle and score a complete image."""
import hashlib,json,time
from pathlib import Path
import torch
from safetensors.torch import load_file
from .components import Vision,Text
from .model import MCJev
from .geometry import Tiling,Admission,TilePlan,iter_views

class Runtime:
    def __init__(self,bundle,device='cpu'):
        root=Path(bundle).resolve();manifest=json.loads((root/'manifest.json').read_text())
        if manifest.get('format')!='shorecrab-a60-inference-v1':raise ValueError('unsupported bundle')
        for name,digest in manifest['files'].items():
            p=(root/name).resolve()
            if not p.is_relative_to(root):raise ValueError('bundle path escape')
            with p.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
            if actual!=digest:raise ValueError('bundle checksum mismatch: '+name)
        if device not in ['cpu','mps','cuda']:raise ValueError('device must be cpu, mps or cuda')
        self.device=device;self.bundle_id=manifest['checkpoint_sha256']
        self.model=MCJev(Vision(),Text(root/'tokenizer'),dim=384,heads=6,global_queries=64,local_queries=32,
            fusion_layers=2,resampler_residual='learned_queries',resampler_bank_std=1.0,resampler_qcond_init='zero',summary_source='overview')
        self.model.load_state_dict(load_file(root/'model.safetensors'),strict=True)
        self.model.eval();self.model.vision.fuse();self.model.requires_grad_(False).to(device)
        self.tiling=Tiling(locals='beyond_canvas');self.admission=Admission()
    @torch.inference_mode()
    def score(self,image,question,choices):
        if not isinstance(question,str) or not question.strip():raise ValueError('question must be non-empty')
        if not isinstance(choices,list) or not 2<=len(choices)<=16:raise ValueError('provide 2–16 choices')
        if any(not isinstance(c,str) or not c.strip() for c in choices):raise ValueError('choices must be non-empty strings')
        if len(set(choices))!=len(choices):raise ValueError('choices must be distinct')
        image=image.convert('RGB');plan=TilePlan(*image.size,self.tiling);self.admission.check(plan)
        start=time.perf_counter();views=list(iter_views(image,self.tiling))
        out=self.model(views,question,choices,encoded=self.model.encode_views(views,fixed_chunk=False))
        if out['processed_tiles']!=plan.count:raise RuntimeError('incomplete image coverage')
        confidence=out['answerability_logit'].sigmoid();p=(out['option_logits'].softmax(-1)*confidence).cpu().tolist();none=float(1-confidence)
        winner=max(range(len(p)+1),key=(p+[none]).__getitem__)
        return {'model':'shorecrab-128m-a60','answer_index':winner if winner<len(p) else None,
            'answer':choices[winner] if winner<len(p) else None,'probabilities':p,'none_probability':none,
            'calibrated':False,'usage':{'local_tiles':plan.count,'input_size':list(image.size)},
            'latency_ms':(time.perf_counter()-start)*1000,'checkpoint_sha256':self.bundle_id}
