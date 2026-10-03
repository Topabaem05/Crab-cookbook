"""Create the configuration consumed by vLLM's standard model loader."""
import argparse,hashlib,json,os,shutil
from pathlib import Path

def prepare_bundle(source,destination):
    source=Path(source).resolve();destination=Path(destination).resolve()
    if destination.exists():raise ValueError('destination must not already exist')
    manifest=json.loads((source/'manifest.json').read_text())
    if manifest.get('format')!='shorecrab-a60-inference-v1':raise ValueError('unsupported bundle')
    for name,digest in manifest['files'].items():
        path=(source/name).resolve()
        if not path.is_relative_to(source):raise ValueError('bundle path escape')
        with path.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
        if actual!=digest:raise ValueError('bundle checksum mismatch: '+name)
    destination.mkdir(parents=True)
    try:os.link(source/'model.safetensors',destination/'model.safetensors')
    except OSError:shutil.copyfile(source/'model.safetensors',destination/'model.safetensors')
    shutil.copytree(source/'tokenizer',destination/'tokenizer')
    config=json.loads((source/'tokenizer/config.json').read_text())
    config.update(architectures=['ShoreCrabA60Model'],dtype='float32',max_position_embeddings=2)
    (destination/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    (destination/'provenance.json').write_text(json.dumps(dict(checkpoint_sha256=manifest['checkpoint_sha256'],weight_sha256=manifest['files']['model.safetensors']),indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',required=True);p.add_argument('--output',required=True);a=p.parse_args();prepare_bundle(a.bundle,a.output)
