"""Two output views of the same A60 candidate distribution; no text generation."""
import math

def probability_view(choices,result):
    p=list(result['probabilities']);none=float(result['none_probability'])
    if len(p)!=len(choices) or not 2<=len(choices)<=16:raise ValueError('candidate count differs')
    values=[float(x) for x in p]+[none]
    if any(not math.isfinite(x) or not 0<=x<=1 for x in values) or not math.isclose(sum(values),1.,abs_tol=1e-4):
        raise ValueError('invalid model probability distribution')
    return dict(choices=list(choices),probabilities=values[:-1],none_probability=none,calibrated=False)

def select_answer(choices,result):
    result=probability_view(choices,result);p=result['probabilities']+[result['none_probability']]
    i=max(range(len(p)),key=p.__getitem__)
    return dict(answer=choices[i] if i<len(choices) else None,answer_index=i if i<len(choices) else None,
        probability=p[i],none_probability=result['none_probability'],calibrated=False)
