from track_gen.utils import clamp

import math
import pygame
import numpy as np
import scipy.spatial
import track_gen.bezier as bezier

class Track():
    POINT_OFFSET =150
    NUM_OFFSET_POINTS = 10
    OFFSET_SCALE_FROM_CENTER = 0.25
    MIN_MIDPOINT_WEIGHTING = 0.5
    MAX_MIDPOINT_WEIGHTING = 0.8
    
    def __init__(self, num_points: int, x_bounds: list[int], y_bounds: list[int], config: dict, seed: int = None):
        self.num_points = num_points
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.seed = seed
        self.rng = np.random.default_rng(seed)   
        self.config = config    
     
        self.points = None
        self.hull_vertices = None
        self.track_vertices = []
        
        self.genotype = []
        
        
        self.offset_vertices = []

        self._bezier =  bezier.Bezier(1, 0.01)
    
                
    def render(self, screen: pygame.Surface):
        for i in range(0, len(self.track_vertices)):
            current_point = i
            next_point = clamp(i + 1, 0, len(self.track_vertices))            
        
            # draw line
            pygame.draw.line(screen, (255,0,0), self.track_vertices[current_point], self.track_vertices[next_point], 1)
            
        # plot the convex hull and its vertices
        for i in range(0, len(self.hull_vertices)):
            current_point = i
            next_point = clamp(i+1, 0, len(self.hull_vertices))
            
            pygame.draw.line(screen, (0,255,0), self.hull_vertices[current_point], self.hull_vertices[next_point], 1)
            pygame.draw.circle(screen, (0,0,255), self.hull_vertices[i], 1, 1)

            

    def generate_track(self):
        # generate a random number of points and get the convex hull
        x_coords = self.rng.uniform(self.x_bounds[0], self.x_bounds[1], self.num_points)
        y_coords = self.rng.uniform(self.y_bounds[0], self.y_bounds[1], self.num_points)
        
        self.points = np.column_stack((x_coords, y_coords))
        # generate a convex hull (border)
        convex_hull = scipy.spatial.ConvexHull(self.points)
        self.hull_vertices = self.points[convex_hull.vertices]
    
        # add a random offset to break up square shape
        for idx, x in np.ndenumerate(self.hull_vertices):
            offsets = self.rng.uniform(-self.config["POINT_OFFSET"], self.config["POINT_OFFSET"], 1)
            self.hull_vertices[idx] += offsets
            
        # randomly offset set amount of points 
        # make sure these points arent neighbours
        for i in range(self.config["NUM_OFFSET_POINTS"]):
            self.hull_vertices = self._offset_random_point(self.hull_vertices)
            
        num_vertices = len(self.hull_vertices)
        
        self.midpoints = []
        
        # apply bezier curves by selection 2 random weighted midpoints
        for i in range(0, num_vertices):
            previous_vertex = clamp((num_vertices - 1) + i, 0, num_vertices)
            current_vertex = i
            next_vertex = clamp(i + 1, 0, num_vertices)
     
            # for each vertex
            # get the previous, current and next
            # find points on the interpolated lines 75* of the way through towards the current point
            # use these are the base of the quadractic bezier
            midpoint_start = self._weighted_midpoint(
                self.hull_vertices[previous_vertex], self.hull_vertices[current_vertex], 
                self.rng.uniform(self.config["MIN_MIDPOINT_WEIGHTING"], self.config["MAX_MIDPOINT_WEIGHTING"])
            )
            midpoint_end = self._weighted_midpoint(
                self.hull_vertices[next_vertex], self.hull_vertices[current_vertex], 
                self.rng.uniform(self.config["MIN_MIDPOINT_WEIGHTING"], self.config["MAX_MIDPOINT_WEIGHTING"])
            )
            weighted_point = self.hull_vertices[current_vertex]
                        
            # calculate the coordinates
            bezier_coordinates = self._bezier.generate_bezier(
                self._bezier.QUADRATIC, 
                [midpoint_start[0], weighted_point[0], midpoint_end[0]], # x weights
                [midpoint_start[1], weighted_point[1], midpoint_end[1]], # y weights
            )
            
            self.track_vertices.extend(bezier_coordinates)

    def _offset_random_point(self, points):
                    
        # for a random vertex, find its midpoint and offset it by a random scale factor
        center = np.mean(points, axis=0)
        index = round(self.rng.random() * len(points) - 2)

    
        point = self._midpoint(points[index], points[(index + 1) % len(points)])
    
        scale_offset = self.rng.uniform(self.config["MIN_OFFSET_MIDPOINT"], self.config["MAX_OFFSET_MIDPOINT"])
        displaced = center[0] + scale_offset * (point[0] - center[0]), center[1] + scale_offset * (point[1] - center[1])
    
    
        return np.insert(points, index+1, displaced, 0)
        
    def _offset_random_point_not(self, points):
        not_neighbour = True
        
        while not_neighbour:
            if len(self.offset_vertices) >= math.floor((len(points) / 2)):
                return points
            
            # for a random vertex, find its midpoint and offset it by a random scale factor
            center = np.mean(points, axis=0)
            index = round(self.rng.random() * len(points) - 2)
        
            if (index in self.offset_vertices):
                continue
            
            if (index + 1) in self.offset_vertices: 
                continue
            
            if (index - 1) in self.offset_vertices:
                continue
        
            point = self._midpoint(points[index], points[(index + 1) % len(points)])
        
            scale_offset = self.rng.uniform(self.config["MIN_OFFSET_MIDPOINT"], self.config["MAX_OFFSET_MIDPOINT"])
            displaced = center[0] + scale_offset * (point[0] - center[0]), center[1] + scale_offset * (point[1] - center[1])
        
            self.offset_vertices.append(index)
        
            return np.insert(points, index+1, displaced, 0)
    
    def _midpoint(self, p1, p2):
        return (
            (p1[0] + p2[0]) / 2, 
            (p1[1] + p2[1]) / 2
        )    
        
    def _weighted_midpoint(self, p1: tuple, p2: tuple, pos: float):
        return (
            ((1- pos) * p1[0]) + (pos * p2[0]),
            ((1- pos) * p1[1]) + (pos * p2[1])
        )