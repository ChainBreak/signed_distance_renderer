import torch
from .shape import Shape

class Sphere(Shape):

    def __init__(self,sphere_radius):
        super().__init__()
        self.sphere_radius = sphere_radius

    def signed_distance_function(self,point_tensor):
        signed_distance = (point_tensor**2).sum(dim=1).sqrt() - self.sphere_radius
        return signed_distance