#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt

from shapes import Compose, Sphere,Cube
from operations import Start, Join, Cut, Intersect

def main():
    draw_frame()

def draw_frame():
    width,height = 640,480
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

        depth_tensor += sign_distance_tensor
    
    return depth_tensor

def global_sign_distance_function(position_tensor):
   
 
    return Compose([
        Start( Cube(cube_size=10, position=(0,0,15), orientation=(0,1.05,0)) ),
        Cut( Sphere(sphere_radius=6, position=(0,0,15))  ),
        Intersect( Sphere(sphere_radius=7, position=(0,0,15))  ),
    ]).compute_signed_distance(position_tensor)

     






if __name__ == "__main__":
    main()
    
