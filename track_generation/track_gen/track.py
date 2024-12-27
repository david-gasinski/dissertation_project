from track_gen.utils import clamp

import scipy
import pygame
import numpy as np
import scipy.spatial
import bezier

class Track():
    POINT_OFFSET = 75
    NUM_OFFSET_POINTS = 5
    OFFSET_SCALE_FROM_CENTER = 0.3
    BEZIER_INTERPOLATION_WEIGHTING = 0.875
    
    def __init__(self, num_points: int, x_bounds: list[int], y_bounds: list[int], seed: int = None):
        self.num_points = num_points
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.rng = np.random.default_rng(seed)    
        
        self.points = None
        self.hull_vertices = None
        self.midpoints = None
        self._bezier =  bezier.Bezier(1, 0.01)
                
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
    
        # randomly offset set amount of points 
        # make sure these points arent neighbours
        for i in range(self.NUM_OFFSET_POINTS):
            self.hull_vertices = self._offset_random_point(self.hull_vertices)
            
        num_vertices = len(self.hull_vertices)
        
        
        # apply bezier curves by selection 2 random weighted midpoints
        for i in range(0, num_vertices):
            previous_vertex = clamp(num_vertices + i, 0, num_vertices)
            current_vertex = i
            next_vertex = clamp(i + 1, 0, num_vertices)
     
            # for each vertex
            # get the previous, current and next
            # find points on the interpolated lines 75* of the way through towards the current point
            # use these are the base of the quadractic bezier
            midpoint_start = self._weighted_midpoint(
                self.hull_vertices[previous_vertex], self.hull_vertices[current_vertex], 
                self.BEZIER_INTERPOLATION_WEIGHTING
            )
            midpoint_end = self._weighted_midpoint(
                self.hull_vertices[next_vertex], self.hull_vertices[current_vertex], 
                self.BEZIER_INTERPOLATION_WEIGHTING
            )
            weighted_point = self.hull_vertices[current_vertex]
            
            # calculate the coordinates
            bezier_coordinates = self._bezier.generate_bezier(
                self._bezier.QUADRATIC, 
                [midpoint_start[0], weighted_point[0], midpoint_end[0]], # x weights
                [midpoint_start[1], weighted_point[1], midpoint_end[1]], # y weights
            )
            
            # remove the current index and replace it with the resultant
            # array of bezier coordinates
            
            
            
            return
        
    def _apply_bezier(self, point_index: int,  wx: list[float], wy:list[float]) -> np.ndarray:
        bezier_coords = self._bezier.generate_bezier(bezier.Bezier.QUADRATIC, wx, wy)
        
        # remove the coordinate of the point and replace it with bezier
        
        return
        
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
        
    def _weighted_midpoint(self, p1: tuple, p2: tuple, pos: float):
        return (
            (1- pos) * p1[0] + pos * p2[0],
            (1- pos) * p1[1] + pos * p2[1]
        )