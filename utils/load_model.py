# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    load_model.py                                      :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: j2457li <j2457li@student.42.fr>            +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/04/17 21:44:33 by j2457li           #+#    #+#              #
#    Updated: 2023/04/17 21:44:39 by j2457li          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from models.alexnet import *
from models.googlenet import *
from models.resnet import *
from models.vgg import *


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
    elif model_name == "vgg11":
        return vgg11_bn()
    elif model_name == "vgg13":
        return vgg13_bn()
    elif model_name == "vgg16":
        return vgg16_bn()
    elif model_name == "vgg19":
        return vgg19_bn()
    elif model_name == "googlenet":
        return GoogleNet()
    else:
        raise ValueError(
            "Sorry, we do not support the model you specified, please try adjusting spelling or change to another model"
        )
