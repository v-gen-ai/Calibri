def get_pipeline_by_name(model_name):
    if model_name == "stabilityai/stable-diffusion-3.5-medium" or model_name == "stabilityai/stable-diffusion-3.5-large":
        from .sd3_sg import SGSD3Pipeline
        return SGSD3Pipeline
    elif model_name == "black-forest-labs/FLUX.1-dev":
        from .flux_sg import SGFluxPipeline
        return SGFluxPipeline
    else:
        raise NameError("unknown model name")
