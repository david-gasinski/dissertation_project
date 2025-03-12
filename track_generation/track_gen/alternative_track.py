import math
import pygame
from dataclasses import dataclass
import numpy as np
import scipy.spatial
import matplotlib.pyplot as plt

@dataclass
class Genotype:
    radial_distance : float
    angular_distance : float
    tangent: float
    
class Track():
    
    def __init__(self, num_points: int, x_bounds: list[int], y_bounds: list[int], config: dict, seed: int = None):
        self.num_points = num_points
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.rng = np.random.default_rng(seed)
        
        self.midpoint = [0,0]
    
    def render(self, screen: pygame.Surface):
        return
        
    def generate_track(self):
        # generate a random number of points and get the convex hull
        x_coords = self.rng.uniform(self.x_bounds[0], self.x_bounds[1], self.num_points)
        y_coords = self.rng.uniform(self.y_bounds[0], self.y_bounds[1], self.num_points)
        
        self.points = np.column_stack((x_coords, y_coords))
        
        self.hull_vertices = self.points
        plt.scatter(self.hull_vertices[::, 0], self.hull_vertices[::, 1])
        plt.show()
        
        # find the radial distance    
        radial_dist = np.sqrt(
            np.power((self.hull_vertices[::,0] - self.midpoint[0]), 2) + 
            np.power((self.hull_vertices[::,1] - self.midpoint[1]), 2)
        )
        
        # find the cosine angle between up vector [0,1] 
        up_vec = np.array([0,1]) 
        up_vec_norm = up_vec / np.linalg.norm(up_vec)

        hull_vertices_norm = self.hull_vertices / np.linalg.norm(self.hull_vertices, axis=1, keepdims=True)
        
        cosine_angles = np.dot(hull_vertices_norm, up_vec_norm)
        inverse_cos = np.arccos(cosine_angles) * (180/math.pi)
        print(inverse_cos)
    
config = {
    "WIDTH": 800, 
    "HEIGHT": 800,
    "POINT_OFFSET": 30,
    "NUM_OFFSET_POINTS": 15,
    "MIN_OFFSET_MIDPOINT": 0.75,
    "MAX_OFFSET_MIDPOINT" : 1,
    "MIN_MIDPOINT_WEIGHTING": 0.3,
    "MAX_MIDPOINT_WEIGHTING": 0.7,
    "TRACK_IMG_FOLDER": "tracks/",
    "NUM_POINTS": 10
}

# create a new track
x_offset = config['WIDTH'] / 4
y_offset = config['HEIGHT'] / 4
track_bounds = (
    [-x_offset, x_offset],
    [-y_offset, y_offset]
)

track = Track(
    config["NUM_POINTS"], track_bounds[0], track_bounds[1], config, 10   
)

track.generate_track()