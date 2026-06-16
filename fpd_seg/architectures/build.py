from .new_mamba_net import New_Mamba_Net


SUPPORTED_MODELS = {
    "New_Mamba_Net": New_Mamba_Net,
}


def build_model(config):
    model_name = config.MODEL.TYPE
    if model_name not in SUPPORTED_MODELS:
        supported = ", ".join(sorted(SUPPORTED_MODELS))
        raise ValueError(f"Unsupported model '{model_name}'. Supported models: {supported}")

    return SUPPORTED_MODELS[model_name](num_classes=2, num_channels=1), False
