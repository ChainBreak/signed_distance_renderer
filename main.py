#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt

def main():
    draw_frame()

def draw_frame():
    width,height = 640,480

    ray_origin_tensor, ray_directon_tensor = get_pixel_rays(width,height,(0,0,0),None)

    depth_tensor =  march_rays_into_scene(ray_origin_tensor, ray_directon_tensor)
    depth_tensor = depth_tensor.clip(0,20)
    depth_tensor = 1/ depth_tensor
    plt.imshow(depth_tensor.reshape(height,width).numpy())
    plt.show()



def get_pixel_rays( width, height, camera_position, camera_rotation):

    number_of_rays = width*height

    x = torch.linspace(-1 , 1 , width )
    y = torch.linspace(-height/width, height/width, height)

    y_grid, x_grid = torch.meshgrid(y,x)

    z_grid = torch.ones_like(y_grid)

    ray_directon_tensor = torch.stack((x_grid, y_grid,z_grid),dim=2).reshape( number_of_rays  ,3)

    ray_directon_tensor = F.normalize(ray_directon_tensor)

    ray_origin_tensor = torch.tensor(camera_position).repeat(number_of_rays,1)

    return ray_origin_tensor, ray_directon_tensor

def march_rays_into_scene(ray_origin_tensor, ray_directon_tensor):
    number_of_rays = ray_directon_tensor.shape[0]

    depth_tensor = torch.zeros(number_of_rays,1)
    
    for i in range(50):
        position_tensor = ray_origin_tensor + depth_tensor * ray_directon_tensor
   
        sign_distance_tensor = global_sign_distance_function(position_tensor)

        depth_tensor += sign_distance_tensor
    
    return depth_tensor

def global_sign_distance_function(position_tensor):
    s1 = sphere_sign_distance_function(position_tensor,[0,0,11])
    s2 = sphere_sign_distance_function(position_tensor,[2,0,10])
    return torch.minimum(s1,s2)

def sphere_sign_distance_function(position_tensor,position):
    sphere_origin = torch.tensor([position])
    sphere_radius = 5

    sign_distance = ((position_tensor-sphere_origin)**2).sum(dim=1).sqrt() - sphere_radius

    
    return sign_distance.reshape(-1,1)


if __name__ == "__main__":
    main()
    
