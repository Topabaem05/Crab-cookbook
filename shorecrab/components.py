"""Image and text components used only for inference."""
import torch
from torch import nn

class Stage(nn.Module):
    def __init__(self,base):
        super().__init__();self.base=base;self.adapter=nn.Identity()
    def forward(self,x):return self.adapter(self.base(x))

class Vision(nn.Module):
    def __init__(self):
        super().__init__()
        import timm
        base=timm.create_model('fastvit_t8',pretrained=False)
        self.stem=base.stem;self.stages=nn.ModuleList([Stage(s) for s in base.stages]);self.channels=(192,384)
    def forward(self,x):
        x=self.stem(x);out=[]
        for i,stage in enumerate(self.stages):
            x=stage(x)
            if i>=2:out.append(x)
        return out
    def fuse(self):
        from timm.utils import reparameterize_model
        for part in [self.stem,*[s.base for s in self.stages]]:reparameterize_model(part,inplace=True)
        return self.requires_grad_(False).eval()

class Text(nn.Module):
    def __init__(self,path):
        super().__init__()
        from transformers import AutoConfig,AutoModel,AutoTokenizer
        self.tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=False)
        self.encoder=AutoModel.from_config(AutoConfig.from_pretrained(path,local_files_only=True),trust_remote_code=False)
        self.projection=nn.Identity();self.max_question=96;self.max_choice=48
    def encode(self,question,choices,device):
        outputs=[]
        for texts,limit in [([question],self.max_question),(choices,self.max_choice)]:
            x=self.tokenizer(texts,padding=True,truncation=False,return_tensors='pt')
            if x['input_ids'].shape[1]>limit:raise ValueError(f'text exceeds {limit} tokens')
            x={k:v.to(device) for k,v in x.items()}
            outputs.append((self.encoder(**x).last_hidden_state,x['attention_mask'].bool()))
        return outputs[0][0],outputs[1][0],outputs[1][1]
