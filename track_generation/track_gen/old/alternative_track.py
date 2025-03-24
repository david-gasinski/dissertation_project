import math
import pygame
from dataclasses import dataclass
import numpy as np
import scipy.spatial

from typing import Union

from track_gen.utils import LinearAlgebra
from track_gen.utils  import clamp
from track_gen.bezier import Bezier as bezier
from track_gen.utils import convert_to_screen, convert_to_screen_scalar


 
@dataclass
class ControlPoint:
    radial_distance : float
    angular_distance : float
    tangent_slope: float
    x: float
    y: float
    
    def get_points_tangent(self, i_min: int, i_max: int, i: int, slope: float = None) -> np.ndarray:
        """
            returns all points in a given interval i from range i_min, i_max. 
            uses x,y as the origin of the line
            
            Optional:
                slope: float
        """
        if (slope == None):
            slope = self.tangent_slope
        
        return LinearAlgebra.line_eq(
            slope, self.x, self.y, i_min, i_max, i
        )
        
    def intersects(self, control_point , i_min: float, i_max: float, use_origin: bool = False) -> Union[np.ndarray, None]:
        """
            Check if line with gradient ```slope``` and origin ```b_x, b_y``` or (0,0 if use_origin) intersects ```control_point``` within the range ```i_min, i_max```
            returns the point of intersection or None
        
        """       
        current_point_interval = LinearAlgebra.line_eq(
            self.tangent_slope, self.x, self.y, i_min, i_max, 2
        )
        
        control_point_interval = LinearAlgebra.line_eq(
            control_point.tangent_slope, control_point.x, control_point.y, i_min, i_max, 2
        )
        
        intersection_point = LinearAlgebra.get_intersect(
            current_point_interval[0], current_point_interval[1], control_point_interval[0], control_point_interval[1]
        )

        return intersection_point
                
class Track():
   
    def __init__(self, num_points: int, x_bounds: list[int], y_bounds: list[int], config: dict, seed: int = None):
        self.num_points = num_points
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.rng = np.random.default_rng(seed)
        self.midpoint = [0,0]
        
        self.config = config
        self.seed = seed
        
        # if any of these are set to false, the track is discarded from the population 
        self.valid_track = True
        self.closed = True
        
        self.genotypes = []
        
        self._bezier = bezier(1, 0.01)
        
    
    def render(self, screen: pygame.Surface):
        # create a new pygame surface object
        
        surface = pygame.Surface((self.config["WIDTH"], self.config["HEIGHT"]))
        
        # for each genotype
        for genotype in range(0, self.num_points):
            print(genotype)
            current = self.genotypes[genotype] 
            
            # clamp the next genotype between 0 and self.num_points
            next_genotype = self.genotypes[clamp(genotype+1,0, self.num_points)] 
            
            # get the point of intersection within the scope of the bounds
            tangent_intersection = current.intersects(next_genotype, self.x_bounds[0], self.x_bounds[1])
            
            if tangent_intersection.any() == float('inf'):
                continue
            
            # calculate the bezier curve using the tangent intersection as the weightings
            # plot
            bezier_coords = self._bezier.generate_bezier(
                bezier.QUADRATIC,
                [current.x, tangent_intersection[0], next_genotype.x],
                [current.y, tangent_intersection[1] ,next_genotype.y ]
            )
            
            pygame.draw.circle(
                surface, (255,0,0), (convert_to_screen_scalar(current.x), convert_to_screen_scalar(current.y)), 5
            )

            pygame.draw.line(
                surface, (0,0,255), (convert_to_screen_scalar(current.x), convert_to_screen_scalar(current.y)),
                (convert_to_screen_scalar(next_genotype.x), convert_to_screen_scalar(next_genotype.y)), 3
            )

            num_coords = len(bezier_coords)
            for coordinate in range(num_coords):
                # clamp each coordinate
                current_coord = coordinate
                next_coord = clamp(coordinate+1, 0, num_coords)
                
                pygame.draw.line(surface, (0, 255, 0), convert_to_screen(bezier_coords[current_coord]), convert_to_screen(bezier_coords[next_coord]), 1)
        
        screen.blit(surface, (0,0))
        
    def generate_track(self):
        # generate a fixed number of points in random locations
        x_coords = self.rng.uniform(self.x_bounds[0], self.x_bounds[1], self.num_points)
        y_coords = self.rng.uniform(self.y_bounds[0], self.y_bounds[1], self.num_points)
        
        # to ensure points are in sequence, take a convex hul first
        self.points = scipy.spatial.ConvexHull(np.column_stack((x_coords, y_coords))).points
        print(self.points)
        
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
        slopes = LinearAlgebra.calculate_slopes(self.points)
        
        # get the slope of the tangents 
        tangents = LinearAlgebra.calculate_slope_tangent(slopes)
    
        # create each control point and append to the genotype    
        for point in range(self.num_points):
            # create a new dataclass
            self.genotypes.append(ControlPoint(
                  radial_dist[point],
                  inverse_cos[point],
                  tangents[point],
                  self.points[point, 0],
                  self.points[point, 1]
        ))
                  