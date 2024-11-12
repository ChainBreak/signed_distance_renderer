import torch
import math

def transformable(sdf_function):

    def wrapper(
            points,
            *args,
            x = 0,
            y = 0,
            z = 0,
            rx = 0,
            ry = 0,
            rz = 0,
            sx = 1,
            sy = 1,
            sz = 1,
            s=1,
            **kwargs,
        ):

        points = translate_points(points, x, y, z)
        points = rotate_points(points, rx, ry, rz)
        points = scale_points(points, s, sx, sy, sz)

        return sdf_function(points, *args, **kwargs)
    return wrapper

def rotate_points(points:torch.Tensor, rx, ry, rz):
    rx = math.radians(rx)
    ry = math.radians(ry)
    rz = math.radians(rz)

    cx = math.cos(rx)
    cy = math.cos(ry)
    cz = math.cos(rz)
    sx = math.sin(rx)
    sy = math.sin(ry)
    sz = math.sin(rz)

    rotation_matrix = torch.tensor(
        [
            [cy*cz, sx*sy*cz - cx*sz, cx*sy*cz + sx*sz],
            [cy*sz, sx*sy*sz + cx*cz, cx*sy*sz - sx*cz],
            [-sy  , sx*cy           , cx*cy           ],
        ]
        )
    rotation_matrix_inverse = rotation_matrix.inverse()

    points = points @ rotation_matrix_inverse.to(points.device)

    return points

def translate_points(points:torch.Tensor, x, y, z):
    points = points - torch.tensor([[x,y,z]],device=points.device)
    return points 

def scale_points(points:torch.Tensor, s, sx, sy, sz):
    return points / torch.tensor([[s*sx, s*sy, s*sz]],device=points.device)