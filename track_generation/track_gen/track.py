import scipy
import pygame
import numpy as np
import scipy.spatial
import math

class Track():
    POINT_OFFSET = 200
    NUM_OFFSET_POINTS = 6
    OFFSET_SCALE_FROM_CENTER = 0.3
    
    def __init__(self, num_points: int, x_bounds: list[int], y_bounds: list[int], seed: int = None):
        self.num_points = num_points
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.rng = np.random.default_rng(seed)    
        
        self.points = None
        self.hull_vertices = None
        self.midpoints = None
        
    def render(self, screen: pygame.Surface):
        #if self.points.any() or self.hull_vertices.any():
        #    return
        
        for point in self.points:
            pygame.draw.circle(screen, (0,0,255),  (point[0], point[1]), 1)
            
        for i in range(0, len(self.hull_vertices)):
            current_point = i
            next_point = i + 1
            
            if next_point > len(self.hull_vertices) - 1:
                next_point = 0    
            
            pygame.draw.line(screen, (255,0,0), self.hull_vertices[current_point], self.hull_vertices[next_point], 1)
            
        for point in self.hull_vertices:
            pygame.draw.circle(screen, (255,0,00),  (point[0], point[1]), 1)
            
        
    def generate_track(self):
        # generate a random number of points and get the convex hull
        x_coords = self.rng.uniform(self.x_bounds[0], self.x_bounds[1], self.num_points)
        y_coords = self.rng.uniform(self.y_bounds[0], self.y_bounds[1], self.num_points)
        
        # add a random offset to break up square shape
        for idx, x in np.ndenumerate(x_coords):
            offsets = self.rng.uniform(-self.POINT_OFFSET, self.POINT_OFFSET, 2)
            x_coords[idx] += offsets[0]
            y_coords[idx] += offsets[1]
        
        self.points = np.column_stack((x_coords, y_coords))
        # generate a convex hull (border)
        convex_hull = scipy.spatial.ConvexHull(self.points)
        self.hull_vertices = self.points[convex_hull.vertices]
    
        # randomly offset set amount of points points
        for i in range(self.NUM_OFFSET_POINTS):
            self.hull_vertices = self._offset_random_point(self.hull_vertices)
        
        
    def _offset_random_point(self, points):
        # for a random vertex, find its midpoint and offset it by a random scale factor
        center = np.mean(points, axis=0)
        self.rng.random()
        index = round(self.rng.random() * len(points) - 2)
        point = self._midpoint(points[index], points[(index + 1) % len(points)])
        
        scale_offset = self.rng.uniform(self.OFFSET_SCALE_FROM_CENTER, 1)
        displaced = center[0] + scale_offset * (point[0] - center[0]), center[1] + scale_offset * (point[1] - center[1])
        
        return np.insert(points, index+1, displaced, 0)
    
    def _midpoint(self, p1, p2):
        return (
            (p1[0] + p2[0]) / 2, 
            (p1[1] + p2[1]) / 2
        )    
        
    def _weighted_midpoint(p1, p2, pos):
        return (
            (1- pos) * p1[0] + pos * p2[0],
            (1- pos) * p1[1] + pos * p2[1]
        )