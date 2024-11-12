
from shapes import cube, sphere
from operations import join, cut, intersect
from transformable import transformable
import math

def signed_distance_function(p,t):
    tilt = 30*math.sin(2*math.pi*t)
    rotation=360*t
    return turn_table(p,t,z=2.5,rx=tilt,ry=rotation)

@transformable
def turn_table(p,t):
    return thing(p,t)

transformable
def thing(p,t):
    d = cube(p,)
    d = intersect(d,sphere(p,s=1.5))
    d= join(d,sphere(p,x=-0.3))
    d_sphere = sphere(p, x=0.6, z=0.6)
    d = cut(d, d_sphere)
    d_sphere = sphere(p, x=0.6, z=-0.6)
    d = cut(d, d_sphere)

    return d