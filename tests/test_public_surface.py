"""Keep the distribution restricted to the documented inference surface."""
import ast
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def test_runtime_has_no_research_pipeline_dependencies():
    forbidden_imports=('mcjev','torch.optim','datasets','wandb','peft','trl')
    forbidden_calls={'backward','fit','train_run','optimizer_groups','save_checkpoint'}
    for path in (ROOT/'shorecrab').rglob('*.py'):
        tree=ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):names=[a.name for a in node.names]
            elif isinstance(node,ast.ImportFrom):names=[node.module or '']
            else:names=[]
            assert not any(n==f or n.startswith(f+'.') for n in names for f in forbidden_imports),path
            if isinstance(node,ast.Call):
                name=node.func.attr if isinstance(node.func,ast.Attribute) else node.func.id if isinstance(node.func,ast.Name) else ''
                assert name not in forbidden_calls,(path,name)

def test_examples_use_the_client_boundary():
    for path in (ROOT/'examples').glob('*.py'):
        tree=ast.parse(path.read_text())
        imports=[n.module or '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        assert 'shorecrab' in imports,path
