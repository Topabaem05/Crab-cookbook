"""vLLM entry point; lazy registration works in spawned worker processes."""
def register():
    from vllm import ModelRegistry
    ModelRegistry.register_model('ShoreCrabA60Model','shorecrab.native.vllm_model:ShoreCrabA60Model')
