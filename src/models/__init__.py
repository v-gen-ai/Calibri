from functools import partial

def get_pipeline_by_name(model_name):
    if model_name == "stabilityai/stable-diffusion-3.5-medium" or model_name == "stabilityai/stable-diffusion-3.5-large":
        from .sd3_sg import SGSD3Pipeline
        return SGSD3Pipeline
    elif model_name == "flux_mlp_attn":
        from .flux_sg_mlp_attn import SGFluxPipeline_MlpAttn
        return SGFluxPipeline_MlpAttn
    elif model_name == "flux_block":
        from .flux_sg_block import SGFluxPipelineBlock
        return SGFluxPipelineBlock
    elif model_name == "black-forest-labs/FLUX.1-dev":
        from .flux_sg import SGFluxPipeline
        return SGFluxPipeline
    elif model_name == "jieliu/SD3.5M-FlowGRPO-GenEval" or model_name == "jieliu/SD3.5M-FlowGRPO-PickScore":
        from .sd3_lora_sg import SGSD3PipelineLORA
        return SGSD3PipelineLORA
    elif model_name == "qwen":
        from .qwen_sg import SGQwenPipelineCFG
        return SGQwenPipelineCFG
    else:
        raise NameError("unknown model name")
