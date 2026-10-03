import pytest

def test_probability_view_preserves_candidate_order_and_none():
    from shorecrab.outputs import probability_view
    result=probability_view(['red','blue'],{'probabilities':[.2,.5],'none_probability':.3})
    assert result['choices']==['red','blue'] and result['probabilities']==[.2,.5]
    assert result['none_probability']==.3 and result['calibrated'] is False

def test_select_includes_none_and_never_forces_an_answer():
    from shorecrab.outputs import select_answer
    assert select_answer(['red','blue'],{'probabilities':[.2,.5],'none_probability':.3})['answer']=='blue'
    result=select_answer(['red','blue'],{'probabilities':[.2,.1],'none_probability':.7})
    assert result['answer'] is None and result['answer_index'] is None and result['probability']==.7

@pytest.mark.parametrize('result',[
    {'probabilities':[.2],'none_probability':.8},
    {'probabilities':[.2,.5],'none_probability':float('nan')},
    {'probabilities':[.8,.5],'none_probability':.3},
])
def test_invalid_outputs_are_rejected(result):
    from shorecrab.outputs import select_answer
    with pytest.raises(ValueError):select_answer(['red','blue'],result)
