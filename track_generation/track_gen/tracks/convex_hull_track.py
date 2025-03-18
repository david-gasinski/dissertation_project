from __future__ import annotations
from typing import TYPE_CHECKING

from track_gen import utils
import pygame

from track_gen.abstract import abstract_track
import numpy as np


class ConvexHullTrack(abstract_track.Track):

    CONTROL_POINTS = None
    
    def __init__(self, control_points: int) -> None:
        super().__init__(control_points)
                
    def render(self, screen):
       
       # if control points not set, don't render
       if np.any(self.CONTROL_POINTS):
           return
       
       for point in range(self._control_points):
            # get the current and next control point
            _current = self.CONTROL_POINTS[point]
            _next = utils.clamp(point + 1, 0, self._control_points)
   
            _current_origin = (_current[0], _current[1])
            _next_origin = (_next[0], _next[1])
   
            # plot each, and join for the time being
            pygame.draw.circle(screen, (255, 0,0), _current_origin, 3)
            pygame.draw.line(screen, (0,255,0), _current_origin, _next_origin, 1)
               
    
    def fitness(self):
        return super().fitness()
    
    def mutate(self, mutation_points = 1):
        return super().mutate(mutation_points)
    
    def get_genotype(self):
        return super().get_genotype()
    
    def encode_control_point(self, index: int, x: float, y: float, slope: float, x_c1: float, y_c1: float, x_c2: float, y_c2: float):
        return super().encode_control_point(index, x, y, slope, x_c1, y_c1, x_c2, y_c2)
    