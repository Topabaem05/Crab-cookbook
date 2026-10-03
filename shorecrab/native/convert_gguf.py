"""Inference-only conversion to the FP32 A60 native GGUF format."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from gguf import GGUFWriter
from ..runtime import Runtime

def convert(bundle,destination):
    torch.set_num_threads(2)
    runtime=Runtime(bundle,'cpu');core=runtime.model
    tensors={k:v.detach().cpu() for k,v in core.state_dict().items()}
    # Fixed affine BatchNorm in the convolutional MLP; no adaptation of weights.
    for si,stage in enumerate(core.vision.stages):
        for bi,block in enumerate(stage.base.blocks):
            pre=f'vision.stages.{si}.base.blocks.{bi}.mlp.conv'
            bn=block.mlp.conv.bn;gain=bn.weight/(bn.running_var+bn.eps).sqrt()
            tensors[pre+'.gain']=gain.detach();tensors[pre+'.offset']=(bn.bias-bn.running_mean*gain).detach()
    for key,value in list(tensors.items()):
        if 'downsample.proj.0.reparam_conv.weight' in key:
            tensors[key+'.even']=value[::2].contiguous();tensors[key+'.odd']=value[1::2].contiguous()
        if key.endswith('self_attn.in_proj_weight') or key.endswith('self_attn.in_proj_bias'):
            suffix='weight' if key.endswith('weight') else 'bias';prefix=key.rsplit('.',1)[0]
            for name,part in zip(('q','k','v'),value.chunk(3,dim=0)):tensors[f'{prefix}.{name}.{suffix}']=part.contiguous()
    writer=GGUFWriter(str(destination),'shorecrab-a60')
    writer.add_string('shorecrab.checkpoint_sha256',runtime.bundle_id)
    writer.add_uint32('shorecrab.format_version',1)
    for key,value in tensors.items():
        if value.is_floating_point():writer.add_tensor(key,value.float().numpy())
    writer.write_header_to_file();writer.write_kv_data_to_file();writer.write_tensors_to_file();writer.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error('output exists; choose a new path')
    convert(a.bundle,a.output)
