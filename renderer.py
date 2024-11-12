import numpy as np
import cv2
from dataclasses import dataclass
from pathlib import Path
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

@dataclass
class SignedDistanceRender():
    width:int
    height:int
    num_frames:int
    base_output_dir:Path
    size_ratio:float=1.0
    max_steps:int=30
    min_depth:float=0.001
    max_depth:float=10
    device:torch.device=torch.device("cuda")

    def render(self,signed_distance_function):

        output_path_generator = OutputPathGenerator(self.base_output_dir)

        self.render_all_frames(signed_distance_function, output_path_generator)

    def render_all_frames(self, signed_distance_function, output_path_generator):

        for frame_index in range(self.num_frames):

            self.render_single_frame(signed_distance_function, output_path_generator, frame_index)

    def render_single_frame(self, signed_distance_function, output_path_generator, frame_index):       

        t = frame_index / self.num_frames

        depth_image = self.get_depth_image_via_ray_marching(signed_distance_function, t)

        depth_image = depth_image.clip(self.min_depth,self.max_depth)
        inv_depth_image = 1 / depth_image

  
        inv_depth_image -= inv_depth_image.min()
        inv_depth_image /= inv_depth_image.max()
        inv_depth_image = inv_depth_image.cpu().numpy() * 255
        inv_depth_image = inv_depth_image.clip(0,255).astype(np.uint8)

        depth_path = output_path_generator.get_depth_path(frame_index)
        cv2.imwrite(str(depth_path), inv_depth_image)
        


    def get_depth_image_via_ray_marching(self, signed_distance_function, t):

        width = int(self.width * self.size_ratio)
        height = int(self.height * self.size_ratio)

        ray_origin_tensor, ray_directon_tensor = self.get_pixel_rays(width,height)

        depth_tensor = self.march_rays_into_scene(ray_origin_tensor, ray_directon_tensor, signed_distance_function, t)

        return depth_tensor.reshape(height,width)
        
    def get_pixel_rays(self, width, height):

        number_of_rays = width*height
        camera_position = (0,0,0)

        x = torch.linspace(-1 , 1 , width ,device=self.device)
        y = torch.linspace(height/width, -height/width, height, device=self.device)

        y_grid, x_grid = torch.meshgrid(y,x)

        z_grid = torch.ones_like(y_grid)

        ray_directon_tensor = torch.stack((x_grid, y_grid,z_grid),dim=2).reshape( number_of_rays  ,3)

        ray_directon_tensor = F.normalize(ray_directon_tensor)

        ray_origin_tensor = torch.tensor(camera_position,device=self.device).repeat(number_of_rays,1)

        return ray_origin_tensor, ray_directon_tensor

    def march_rays_into_scene(self, ray_origin_tensor, ray_directon_tensor, signed_distance_function, t):

        number_of_rays = ray_directon_tensor.shape[0]

        depth_tensor = torch.zeros(number_of_rays,1,device=self.device)
        
        for i in range(50):
            position_tensor = ray_origin_tensor + depth_tensor * ray_directon_tensor
    
            sign_distance_tensor = signed_distance_function(position_tensor, t)

            depth_tensor += sign_distance_tensor.reshape(-1,1)
        
        return depth_tensor



class OutputPathGenerator():
    def __init__(self,base_dir):
           
        self.versioned_output_dir = self.find_next_version_path(Path(base_dir) )
            
    def find_next_version_path(self, path: Path) -> Path:
        previous_version_numbers = [ int(p.name[1:]) for p in path.glob("v*")] + [0]
        next_version_number = max(previous_version_numbers)+1
        return path / f"v{next_version_number:03}"
    
    def get_depth_path(self, frame_index):
        path = self.versioned_output_dir / "depth" / f"frame_{frame_index:04}.png"

        path.parent.mkdir(exist_ok=True,parents=True)
        return path

    