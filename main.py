#!/usr/bin/env python3
import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt

from shapes import cube, sphere
from operations import join, cut, intersect
from transformable import transformable

def main():
    draw_frame()

def draw_frame():
    width,height = 1920,1080
    device = torch.device("cuda")

    ray_origin_tensor, ray_directon_tensor = get_pixel_rays(width,height,(0,0,0),None,device)

    depth_tensor =  march_rays_into_scene(ray_origin_tensor, ray_directon_tensor)
    depth_tensor = depth_tensor.clip(0,30)
    depth_tensor = 1/ depth_tensor
    plt.imshow(depth_tensor.reshape(height,width).cpu().numpy())
    plt.show()



def get_pixel_rays( width, height, camera_position, camera_rotation,device):

    number_of_rays = width*height

    x = torch.linspace(-1 , 1 , width ,device=device)
    y = torch.linspace(height/width, -height/width, height, device=device)

    y_grid, x_grid = torch.meshgrid(y,x)

    z_grid = torch.ones_like(y_grid)

    ray_directon_tensor = torch.stack((x_grid, y_grid,z_grid),dim=2).reshape( number_of_rays  ,3)

    ray_directon_tensor = F.normalize(ray_directon_tensor)

    ray_origin_tensor = torch.tensor(camera_position,device=device).repeat(number_of_rays,1)

    return ray_origin_tensor, ray_directon_tensor

def march_rays_into_scene(ray_origin_tensor, ray_directon_tensor):
    device = ray_origin_tensor.device

    number_of_rays = ray_directon_tensor.shape[0]

    depth_tensor = torch.zeros(number_of_rays,1,device=device)
    
    for i in range(50):
        position_tensor = ray_origin_tensor + depth_tensor * ray_directon_tensor
   
        sign_distance_tensor = global_sign_distance_function(position_tensor)

        depth_tensor += sign_distance_tensor.reshape(-1,1)
    
    return depth_tensor

def global_sign_distance_function(p):
    t = 1
    d_cube = cube(p, z=3, ry=60, rz=1)
    d_sphere = sphere(p, z=3, x=0.5)
    d = join(d_cube, d_sphere)
    d = cut(d, sphere(p, z=3, x=-0.5))

    return d
                

if __name__ == "__main__":
    main()
    
