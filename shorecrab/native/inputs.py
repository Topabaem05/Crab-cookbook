"""Deterministic, weight-free input preparation shared by native engines."""
import torch
from torch.nn import functional as F
from transformers import AutoTokenizer
from ..geometry import Tiling,Admission,TilePlan,iter_views,feature_geometry,geometry_embedding

FIELDS=('pixels','owned_weights','geometry','question_ids','question_mask','choice_ids','choice_mask')

def prepare(image,question,choices,tokenizer_path):
    if not isinstance(question,str) or not question.strip():raise ValueError('non-empty question required')
    if not isinstance(choices,list) or not 2<=len(choices)<=16 or any(not isinstance(c,str) or not c.strip() for c in choices) or len(set(choices))!=len(choices):
        raise ValueError('2–16 distinct, non-empty choices required')
    image=image.convert('RGB');tiling=Tiling(locals='beyond_canvas');Admission().check(TilePlan(*image.size,tiling))
    views=list(iter_views(image,tiling));tokenizer=AutoTokenizer.from_pretrained(tokenizer_path,local_files_only=True,trust_remote_code=False)
    q=tokenizer([question],return_tensors='pt',padding=True,truncation=False)
    a=tokenizer(choices,return_tensors='pt',padding=True,truncation=False)
    if q['input_ids'].shape[1]>96 or a['input_ids'].shape[1]>48:raise ValueError('text exceeds A60 token limit')
    return dict(pixels=torch.stack([v.pixels for v in views]),
        owned_weights=torch.stack([F.adaptive_avg_pool2d(v.owned.float(),(32,32)).reshape(-1) for v in views]),
        geometry=torch.stack([geometry_embedding(feature_geometry(v,32,32),384) for v in views]),
        question_ids=q['input_ids'],question_mask=q['attention_mask'],choice_ids=a['input_ids'],choice_mask=a['attention_mask'])
