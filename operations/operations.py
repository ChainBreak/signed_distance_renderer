import torch


def join(*args):
    return torch.minimum(*args)

def cut(body, *tools):
    tools = [-tool for tool in tools]
    return torch.maximum(body,*tools)

def intersect(*distances):
    out_distance = distances[0]
    for distance in distances[1:]:
        out_distance = torch.maximum(out_distance, distance)
    return out_distance

# class Join(Operation):
#     def __init__(self,shape):
#         self.shape = shape

#     def modify_signed_distance(self, position_tensor, current_signed_distance_tensor):
#         new_signed_distance_tensor = self.shape.compute_signed_distance(position_tensor)
#         return torch.minimum(new_signed_distance_tensor,current_signed_distance_tensor)

# class Cut(Operation):
#     def __init__(self,shape):
#         self.shape = shape

#     def modify_signed_distance(self, position_tensor, current_signed_distance_tensor):
#         new_signed_distance_tensor = self.shape.compute_signed_distance(position_tensor)
#         return torch.maximum(-new_signed_distance_tensor,current_signed_distance_tensor)

# class Intersect(Operation):
#     def __init__(self,shape):
#         self.shape = shape

#     def modify_signed_distance(self, position_tensor, current_signed_distance_tensor):
#         new_signed_distance_tensor = self.shape.compute_signed_distance(position_tensor)
#         return torch.maximum(new_signed_distance_tensor,current_signed_distance_tensor)