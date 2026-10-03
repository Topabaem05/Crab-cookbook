# SPDX-License-Identifier: Apache-2.0
# Processor pattern adapted from vLLM's Terratorch integration (vLLM/IBM, 2025).
"""Registered vLLM model: all A60 weights and forward execute in the worker."""
from pathlib import Path
import torch
from torch import nn
from transformers import BatchFeature
from vllm.inputs import mm_input
from vllm.model_executor.layers.pooler import IdentityPooler
from vllm.model_executor.models.interfaces import IsAttentionFree,SupportsMultiModal
from vllm.model_executor.models.interfaces_base import attn_type
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig,MultiModalKwargsItems,PlaceholderRange
from vllm.multimodal.parse import DictEmbeddingItems,MultiModalDataParser
from vllm.multimodal.processing import BaseDummyInputsBuilder,BaseMultiModalProcessor,BaseProcessingInfo
from ..components import Vision,Text
from ..model import MCJev
from .inputs import FIELDS
from .tensor_model import forward


def fields(data,is_shared=True):
    return {k:(MultiModalFieldConfig.shared('image',batch_size=1) if is_shared else MultiModalFieldConfig.batched('image')) for k in FIELDS}

class Parser(MultiModalDataParser):
    def _parse_image_data(self,data):
        if isinstance(data,dict):return DictEmbeddingItems(data,modality='image',required_fields=set(FIELDS),fields_factory=fields)
        raise ValueError('use shorecrab.native.inputs.prepare for A60 image/question/choices')

class Info(BaseProcessingInfo):
    def get_data_parser(self):return Parser(expected_hidden_size=self._get_expected_hidden_size())
    def get_supported_mm_limits(self):return {'image':1}

class Dummy(BaseDummyInputsBuilder):
    def get_dummy_text(self,mm_counts):return ''
    def get_dummy_mm_data(self,seq_len,mm_counts,mm_options):
        return {'image':dict(pixels=torch.zeros(1,3,512,512),owned_weights=torch.ones(1,1024),geometry=torch.zeros(1,1024,384),question_ids=torch.ones(1,2,dtype=torch.long),question_mask=torch.ones(1,2,dtype=torch.long),choice_ids=torch.ones(2,2,dtype=torch.long),choice_mask=torch.ones(2,2,dtype=torch.long))}

class Processor(BaseMultiModalProcessor):
    requires_tokenizer=False
    def _get_mm_fields_config(self,hf_inputs,hf_processor_mm_kwargs,*,is_shared=True):return fields(hf_inputs,is_shared)
    def _get_prompt_updates(self,*args,**kwargs):return []
    def apply(self,inputs,timing_ctx):
        with timing_ctx.record('apply_hf_processor'):
            raw=self._get_hf_mm_inputs(inputs.mm_data_items,inputs.hf_processor_mm_kwargs).passthrough_data
            processed=BatchFeature({k:torch.as_tensor(v).unsqueeze(0) for k,v in raw.items()},tensor_type='pt')
        kwargs=MultiModalKwargsItems.from_hf_inputs(processed,fields(processed,False))
        hashes=inputs.get_mm_hashes(self.info.model_id,self.info.ctx.get_mm_config().mm_hasher_algorithm)
        return mm_input(prompt_token_ids=[1],mm_kwargs=kwargs,mm_hashes=hashes,mm_placeholders={'image':[PlaceholderRange(offset=0,length=0)]})

@attn_type('attention_free')
@MULTIMODAL_REGISTRY.register_processor(Processor,info=Info,dummy_inputs=Dummy)
class ShoreCrabA60Model(nn.Module,IsAttentionFree,SupportsMultiModal):
    supports_multimodal_raw_input_only=True
    is_pooling_model=True
    @classmethod
    def get_placeholder_str(cls,modality,i):
        if modality=='image':return None
        raise ValueError('only image inputs are supported')
    def __init__(self,vllm_config,prefix=''):
        super().__init__()
        if vllm_config.model_config.dtype!=torch.float32:raise ValueError('A60 native validation requires float32')
        if vllm_config.parallel_config.tensor_parallel_size!=1:raise ValueError('A60 currently requires tensor_parallel_size=1')
        if vllm_config.scheduler_config.max_num_seqs!=1:raise ValueError('A60 currently requires max_num_seqs=1')
        root=Path(vllm_config.model_config.model)
        self.core=MCJev(Vision(),Text(root/'tokenizer'),resampler_residual='learned_queries',resampler_bank_std=1.0,resampler_qcond_init='zero',summary_source='overview')
        self.pooler=IdentityPooler()
    def embed_input_ids(self,input_ids,multimodal_embeddings=None,*,is_multimodal=None):
        return torch.zeros((input_ids.shape[0],384),dtype=torch.float32,device=input_ids.device)
    def forward(self,input_ids,positions,intermediate_tensors=None,inputs_embeds=None,**kwargs):
        data={k:kwargs[k][0] for k in FIELDS}
        logits,u=forward(self.core,data)
        confidence=u.sigmoid();p=logits.softmax(-1)*confidence
        # Fixed output width: K probabilities, zero padding, None, then K.
        result=p.new_zeros(18);result[:len(p)]=p;result[16]=1-confidence;result[17]=len(p)
        return result[None]
    def load_weights(self,weights):
        state=dict(weights);self.core.load_state_dict(state,strict=True)
        self.core.eval();self.core.vision.fuse();self.core.requires_grad_(False)
        return set(dict(self.named_parameters()))
