from models.resnet import *
from models.alexnet import AlexNet


def load_model(model_name):
    if model_name == "resnet18":
        return resnet18()
    elif model_name == "resnet34":
        return resnet34()
    elif model_name == "resnet50":
        return resnet50()
    elif model_name == "resnet152":
        return resnet152()
    elif model_name == "resnet101":
        return resnet101()
    elif model_name == "alexnet":
        return AlexNet()
    else:
        raise ValueError("Sorry, we do not support the model you specified, please try adjusting spelling or change to "
                         "another "
                         "model")
