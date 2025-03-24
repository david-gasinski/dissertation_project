from __future__ import annotations
from typing import TYPE_CHECKING

from track_gen import utils
import pygame

from track_gen.abstract import abstract_track
from track_gen import bezier
import numpy as np


class ConvexHullTrack(abstract_track.Track):

    CONTROL_POINTS = None
    
    def __init__(self, control_points: int) -> None:
        # init bezier class
        super().__init__(control_points)
                
    def render(self, screen):
        colour_index = 0
        colours = [
            (49,127,67), (100,107,99), (73,126,118), (106,95,49) ,(217,80,48), (236,124,38), (254,0,0), (47, 69, 56), (96, 111, 140), (245,208,51)
            ]
        
        for control_point in self.CONTROL_POINTS:
            c1 = (self.convert_to_screen(control_point[3]), self.convert_to_screen(control_point[4]))
            c2 = (self.convert_to_screen(control_point[5]), self.convert_to_screen(control_point[6]))
    
            pygame.draw.circle(screen, colours[colour_index], c1, 3)
            pygame.draw.circle(screen, colours[colour_index], c2, 3)
        
            pygame.draw.line(screen, colours[colour_index], c1, c2, 1)
        
            colour_index += 1;

        num_coords = len(self.BEZIER_COORDINATES)
        # render bezier
        for coordinate in range(num_coords):
            current_coord = self.BEZIER_COORDINATES[coordinate]
            next_coord = self.BEZIER_COORDINATES[utils.clamp(coordinate + 1, 0, num_coords)]
    
            screen_coord_current = (self.convert_to_screen(current_coord[0]), self.convert_to_screen(current_coord[1]))            
            screen_coord_next = (self.convert_to_screen(next_coord[0]), self.convert_to_screen(next_coord[1]))
            
            #pygame.draw.circle(screen, (0,255,0), screen_coord_current, 5)
            pygame.draw.line(screen, (0,255,0), screen_coord_current, screen_coord_next, 1)
            
    def calculate_bezier(self):
        control_points = self.get_genotype()
        reserved_control_points = []
        self.BEZIER_COORDINATES = []
        
        _bezier = bezier.Bezier(1, 0.001)
        
        for point in range(self._control_points):
            _current = control_points[point]
            _next = control_points[utils.clamp(point + 1, 0, self._control_points)]
                                
            # calculate the distances and the weightings
            _current_c1 = utils.LinearAlgebra.euclidean_distance([_current[0], _current[1]], [_next[3], _next[4]]) 
            _current_c2 = utils.LinearAlgebra.euclidean_distance([_current[0], _current[1]], [_next[5], _next[6]]) 
            _next_c1 = utils.LinearAlgebra.euclidean_distance([_next[0], _next[1]], [_current[3], _current[4]]) 
            _next_c2 = utils.LinearAlgebra.euclidean_distance([_next[0], _next[1]], [_current[5], _current[6]]) 
                    
            c1_x = _current[3] if _next_c1 <_next_c2 else _current[5]
            c1_y = _current[4] if _next_c1 < _next_c2 else _current[6]
            
            c2_x = _next[3] if _current_c1 < _current_c2 else _next[5]
            c2_y = _next[4] if _current_c1 < _current_c2 else _next[6]
            
            # before generating, check if control points are used by any previous curves
            # this stops spade-like corners forming
            
            # if c1 used already, switch to other point
            if  [c1_x, c1_y] in reserved_control_points:
                c1_x = _current[3] if not _next_c1 <_next_c2 else _current[5]
                c1_y = _current[4] if not _next_c1 < _next_c2 else _current[6]
            # if c2 used, switch to other point as well
            elif [c2_x, c2_y] in reserved_control_points:
                c2_x = _next[3] if not _current_c1 < _current_c2 else _next[5]
                c2_y = _next[4] if not _current_c1 < _current_c2 else _next[6]
            
            # add both points to reserved_control_points 
            reserved_control_points.append([c1_x, c1_y]) 
            reserved_control_points.append([c2_x, c2_y])
            
            # define weights based on the chosen control point
            weights_x = [_current[0], c1_x, c2_x, _next[0]]
            weights_y = [_current[1], c1_y, c2_y, _next[1]]
            
            bezier_coords = _bezier.generate_bezier(
                _bezier.CUBIC,
                weights_x,
                weights_y
            )
            
            self.BEZIER_COORDINATES.extend(bezier_coords) 
        
        # NOW do overlap test for the entire bezier
        # if any, swap c1 again
        #overlap = utils.LinearAlgebra.intersection_bezier_curve(self.BEZIER_COORDINATES)
        #print(f"Bezier curve {point} and {point+1} overlaps: {overlap}")            
          

    def fitness(self):
        return super().fitness()
    
    def mutate(self, mutation_points = 1):
        return super().mutate(mutation_points)
    
    def get_genotype(self):
        return super().get_genotype()
    
    def encode_control_point(self, index: int, x: float, y: float, slope: float, x_c1: float, y_c1: float, x_c2: float, y_c2: float):
        return super().encode_control_point(index, x, y, slope, x_c1, y_c1, x_c2, y_c2)
