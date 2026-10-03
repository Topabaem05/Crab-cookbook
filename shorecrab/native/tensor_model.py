"""Complete A60 forward on tensors; executed inside the vLLM worker."""
import math
import torch

def forward(core,data):
    q=core.text.encoder(input_ids=data['question_ids'],attention_mask=data['question_mask']).last_hidden_state
    a=core.text.encoder(input_ids=data['choice_ids'],attention_mask=data['choice_mask']).last_hidden_state
    qcls=q.mean(1);amask=data['choice_mask'].bool();zs=[];summary=None
    # Bound activation memory independently of the number of native image tiles.
    for vi in range(data['pixels'].shape[0]):
        tokens,_=core.encode_view(data['pixels'][vi:vi+1])
        weights=data['owned_weights'][vi]
        memory=tokens+data['geometry'][vi:vi+1]
        zs.append(core.resampler(memory,weights,qcls,64 if vi==0 else 32))
        if vi==0:summary=(tokens.float()*weights[None,:,None]).sum(1)/weights.sum().clamp_min(1e-12)
    visual=torch.cat(zs,1)+core.memory_type[0]
    memory=torch.cat((visual,q+core.memory_type[1]),1)
    for block in core.fusion:a=block(a,amask,memory)
    h=a[:,0];logits=core.scorer(h).squeeze(-1).float()
    joined=torch.cat((qcls,summary.to(q.dtype),h.mean(0,keepdim=True),h.max(0,keepdim=True).values,q.new_tensor([[math.log(h.shape[0])]])),1)
    u=core.answerability(joined).reshape(()).float()
    return logits,u
