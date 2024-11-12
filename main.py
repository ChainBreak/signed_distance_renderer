#!/usr/bin/env python3


from renderer import SignedDistanceRender
from scenes.example import signed_distance_function

def main():
    renderer = SignedDistanceRender(
        width=1920,
        height=1080,
        size_ratio=1.0,
        max_depth=5,
        num_frames=120,
        base_output_dir="/home/thomas/Videos/signed_distance_renders/"
        
    )

    renderer.render(signed_distance_function)

if __name__ == "__main__":
    main()
    
