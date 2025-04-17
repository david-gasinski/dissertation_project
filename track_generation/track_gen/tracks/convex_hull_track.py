from __future__ import annotations
from typing import TYPE_CHECKING

from track_gen import utils
import pygame

from track_gen.abstract import abstract_track

class ConvexHullTrack(abstract_track.Track):
    
    def __init__(self, control_points: int, seed: int) -> None:
        # init bezier class
        self._fitness = 100
        super().__init__(control_points, seed)
                
    def render(self, screen):
        num_coords = len(self.TRACK_COORDS)
        
        for coordinate in range(num_coords):
            current_coord = self.TRACK_COORDS[coordinate]
            next_coord = self.TRACK_COORDS[utils.clamp(coordinate + 1, 0, num_coords)]
    
            screen_coord_current = (self.convert_to_screen(current_coord[0]), self.convert_to_screen(current_coord[1]))            
            screen_coord_next = (self.convert_to_screen(next_coord[0]), self.convert_to_screen(next_coord[1]))
            
            pygame.draw.line(screen, (0,255,0), screen_coord_current, screen_coord_next, 1)
            
    def fitness(self):
        self._fitness = 100 - (5 * utils.LinearAlgebra.intersection_bezier_curve(self.TRACK_COORDS))
        return self._fitness
    
    