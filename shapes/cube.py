import torch
from .shape import Shape

class Cube(Shape):

    def __init__(self,cube_size):
        super().__init__()
        self.cube_size = cube_size

    def signed_distance_function(self,point_tensor):
        signed_distance_in_each_axis = point_tensor.absolute() - self.cube_size/2
        signed_distance,_ = signed_distance_in_each_axis.max(dim=1)
        return signed_distance