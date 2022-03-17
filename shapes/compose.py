import torch
from .shape import Shape

class Compose(Shape):
    def __init__(self,list_of_oparations):
        super().__init__()
        self.list_of_oparations = list_of_oparations

    def signed_distance_function(self, position_tensor):
        signed_distance_tensor = None
        for operation in self.list_of_oparations:
            signed_distance_tensor = operation.modify_signed_distance(position_tensor,signed_distance_tensor)

        return signed_distance_tensor