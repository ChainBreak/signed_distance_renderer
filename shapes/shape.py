import torch
import math
class Shape():

    def __init__(self,position:tuple=(0.,0.,0.), orientation:tuple=(0.,0.,0.)):
        self.set_position(*position)
        self.set_orientation(*orientation)


    def set_position(self,x,y,z):
        self.position = torch.tensor([[x,y,z]])

    def set_orientation(self,rx,ry,rz):

        cx = math.cos(rx)
        cy = math.cos(ry)
        cz = math.cos(rz)
        sx = math.sin(rx)
        sy = math.sin(ry)
        sz = math.sin(rz)

        self.rotation_matrix = torch.tensor(
            [
                [cy*cz, sx*sy*cz - cx*sz, cx*sy*cz + sx*sz],
                [cy*sz, sx*sy*sz + cx*cz, cx*sy*sz - sx*cz],
                [-sy  , sx*cy           , cx*cy           ],
            ]
            )
        self.rotation_matrix_inverse = self.rotation_matrix.inverse()

    def compute_signed_distance(self,point_tensor):
        point_tensor = self._transform_points_to_local_frame(point_tensor)
        signed_distance_tensor = self.signed_distance_function(point_tensor)
        return signed_distance_tensor.reshape(-1,1)

    def _transform_points_to_local_frame(self,point_tensor):
        # undo the shift
        point_tensor = point_tensor - self.position
        # undo the rotation
        point_tensor = point_tensor @ self.rotation_matrix_inverse

        return point_tensor

    def signed_distance_function(self,point_tensor):
        raise NotImplementedError()