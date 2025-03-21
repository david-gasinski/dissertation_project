from __future__ import annotations
from typing import TYPE_CHECKING

import numpy

import pygame

class Bezier:
    PASCAL = [
             [1],
            [1,1],          
           [1,2,1],         
          [1,3,3,1],        
         [1,4,6,4,1],       
        [1,5,10,10,5,1],    
       [1,6,15,20,15,6,1]] 
    
    LINE_THICKNESS = 1
    CUBIC = 1
    QUADRATIC = 0
    
    QUADRATIC_RATIOS = [1, 1, 1]
    CUBIC_RATIOS = [ 1, 0.5,0.5,1 ]
    
    def __init__(self, id: int, interval: float = 0.01) -> None:
        self.id = id
        self.interval = interval
    
    # used for n bezier curves, implemented although its not used
    def _binomial(self, n: int, k: int) -> int:
        while (n >= len(self.PASCAL)):
            rows = len(self.PASCAL)
            new_row = [1]            
            # add new row if not present
            prev = rows-1
            for i in range(1, rows):  
                new_row.append(self.PASCAL[prev][i-1] + self.PASCAL[prev][i])   
            new_row.append(1)
            self.PASCAL.append(new_row)
        return self.PASCAL[n][k]

    
    def _n_bezier_(self, n: int, t: float, w: list[float]):
        sum = 0
        for k in range (0, n):
            sum += w[k] * self._binomial(n,k) * (1-t)**(n-k) * (t**k)
        return sum

    def _cubic_bezier(self, t: float, w: list[float]) -> float:
        mt = 1-t
        rational = (self.CUBIC_RATIOS[0] * mt**3) + (self.CUBIC_RATIOS[1] * 3 * mt** 2 * t) + (self.CUBIC_RATIOS[2] * 3 * mt*t**2) + (self.CUBIC_RATIOS[3] *t**3)
        return ((w[0] * mt**3) + (w[1] * 3 * mt** 2 * t) + (w[2] * 3 * mt*t**2) + (w[3] *t**3))
    
    def _quadratic_bezier(self, t: float, w: list[float]) -> float:
        mt = 1-t
        rational = (self.QUADRATIC_RATIOS[0] * mt**2) + (self.QUADRATIC_RATIOS[1]*2*mt*t) + (self.QUADRATIC_RATIOS[2] * t**2)
        return ((w[0] * mt**2) + (w[1]*2*mt*t) + (w[2] * t**2)) 
              
    def _n_bezier(self, curve_type: int, wx: list[float], wy: list[float], t: float) -> list[float]:
        if curve_type == self.CUBIC:
            return [
                self._cubic_bezier(t, wx),
                self._cubic_bezier(t, wy)    
            ]
        elif curve_type == self.QUADRATIC:
            return [
                self._quadratic_bezier(t, wx),
                self._quadratic_bezier(t, wy)
            ]
        return [
            self._n_bezier_(curve_type, t, wx),
            self._n_bezier_(curve_type, t, wy)
        ]
              
    def generate_bezier(self, curve_type: int,  wx: list[float], wy: list[float]) -> list[list[float]]:
        curve_coordinates = []
        t = 0
        
        while t <= 1:
            curve_coordinates.append(self._n_bezier(curve_type, wx, wy, t))
            t+= self.interval
        return curve_coordinates
    
    def render_bezier(self, bezier_coords: list[tuple], screen: pygame.Surface) -> None:
        num_coords = len(bezier_coords)
        for i in range(0, num_coords):
            current_point = i
            next_point = i + 1
            
            # don't joint the last two points
            if next_point > num_coords - 1:
                continue   
            
            pygame.draw.line(screen, (0,255,0), bezier_coords[current_point], bezier_coords[next_point], self.LINE_THICKNESS)
            
            