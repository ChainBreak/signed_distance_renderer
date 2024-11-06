import torch
import math

from engine.transformable import transformable

@transformable
def cube(point_tensor):
    signed_distance_in_each_axis = point_tensor.absolute() - 1/2
    signed_distance,_ = signed_distance_in_each_axis.max(dim=1, keepdim=True)
    return signed_distance

@transformable
def sphere(point_tensor):
    signed_distance = (point_tensor**2).sum(dim=1, keepdim=True).sqrt() - 1/2
    return signed_distance