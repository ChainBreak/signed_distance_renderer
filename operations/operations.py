import torch


class Operation():
    def __call__(self, position_tensor, current_signed_distance_tensor):
        return self.modify_signed_distance(position_tensor, current_signed_distance_tensor)

    def modify_signed_distance(self,position_tensor, current_signed_distance_tensor):
        raise NotImplementedError

class Start(Operation):
    def __init__(self,shape):
        self.shape = shape

    def modify_signed_distance(self, position_tensor, current_signed_distance_tensor):
        return self.shape.compute_signed_distance(position_tensor)

class Join(Operation):
    def __init__(self,shape):
        self.shape = shape

    def modify_signed_distance(self, position_tensor, current_signed_distance_tensor):
        new_signed_distance_tensor = self.shape.compute_signed_distance(position_tensor)
        return torch.minimum(new_signed_distance_tensor,current_signed_distance_tensor)

class Cut(Operation):
    def __init__(self,shape):
        self.shape = shape

    def modify_signed_distance(self, position_tensor, current_signed_distance_tensor):
        new_signed_distance_tensor = self.shape.compute_signed_distance(position_tensor)
        return torch.maximum(-new_signed_distance_tensor,current_signed_distance_tensor)

class Intersect(Operation):
    def __init__(self,shape):
        self.shape = shape

    def modify_signed_distance(self, position_tensor, current_signed_distance_tensor):
        new_signed_distance_tensor = self.shape.compute_signed_distance(position_tensor)
        return torch.maximum(new_signed_distance_tensor,current_signed_distance_tensor)