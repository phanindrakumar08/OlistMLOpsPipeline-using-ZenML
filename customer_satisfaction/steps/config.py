from zenml.models import BaseZenModel as BaseParameters

class ModelNameConfig(BaseParameters):
    """
    Model Configs
    """
    model_name: str = "LinearRegression"
    