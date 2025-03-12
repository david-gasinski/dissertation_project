import math
import pygame
from dataclasses import dataclass
import numpy as np
import scipy.spatial
import matplotlib.pyplot as plt

from utils import LinearAlgebra

@dataclass
class ControlPointTangent:
    m_x: float
    m_y: float

@dataclass
class ControlPoint:
    radial_distance : float
    angular_distance : float
    tangent: ControlPointTangent
    x: float
    y: float
    
    
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
        # generate a fixed number of points in random locations
        x_coords = self.rng.uniform(self.x_bounds[0], self.x_bounds[1], self.num_points)
        y_coords = self.rng.uniform(self.y_bounds[0], self.y_bounds[1], self.num_points)
        
        self.points = np.column_stack((x_coords, y_coords))

        plt.scatter(self.points[::, 0], self.points[::, 1])
        plt.show()
        
        # find the radial distance    
        radial_dist = np.sqrt(
            np.power((self.points[:,0] - self.midpoint[0]), 2) + 
            np.power((self.points[:,1] - self.midpoint[1]), 2)
        )
        
        # find the cosine angle between up vector [0,1] 
        up_vec = np.array([0,1]) 
        up_vec_norm = up_vec / np.linalg.norm(up_vec)

        hull_vertices_norm = self.points / np.linalg.norm(self.points, axis=1, keepdims=True)
        
        cosine_angles = np.dot(hull_vertices_norm, up_vec_norm)
        
        # angular distances
        inverse_cos = np.arccos(cosine_angles) * (180/math.pi)
        
        # find the slopes of each point 
        slopes = self.points[:, 1] / self.points[:, 0]
        
        line_eq = LinearAlgebra.line_eq(
            slopes[0], self.points[0,0],  -100, 100, 200
        )
        
        # get slope inverse
        slopes = - (1 / slopes)

        line_eq_tan = LinearAlgebra.line_eq_tan(
            slopes[0], self.points[0,0], self.points[0,1], -100, 100, 200
        )
        
        point_normals = LinearAlgebra.line_eq_tan_np(
            slopes, self.points[:,0 ], self.points[:, 1], -100, 100, 200
        )
        
        # Plot each line
        plt.figure(figsize=(8, 8))
        for i in range(point_normals.shape[1]):
            plt.plot(point_normals[:, 0], point_normals[:, 1], label=f'Line {i+1}')
            plt.scatter(self.points[i, 0], self.points[i, 1])
            plt.plot(line_eq[:, 0], line_eq[:, 1])
        plt.scatter(0,0)
                        
        # plt.scatter(self.points[0, 0], self.points[0, 1])
        # plt.plot(line_eq_tan[:, 0], line_eq_tan[:, 1])
        
        # Add labels, title, and legend
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Plotting Multiple Lines from a NumPy Array')
        plt.legend()
        plt.grid(True)
        # Show the plot
        plt.show()
                    
        
          
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