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
    
    CUBIC = 3
    QUADRATIC = 2

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
        return ((w[0] * mt**3) + (w[1] * 3 * mt** 2 * t) + (w[2] * 3 * mt*t**2) + (w[3] *t**3))
    
    def _quadratic_bezier(self, t: float, w: list[float]) -> float:
        mt = 1-t
        return ((w[0] * mt**2) + (w[1]*2*mt*t) + (w[2] * t**2)) 
    
    
    def _calculate_derivative_weights(w: list[float], second_derivative: bool = False) -> list[float]:
        """
        Calculates the new weights for first and second derivatives of cubic bezier curves

        Args:
            w (list[float]): 4 weights used for cubic beziers

        Returns:
            list[float]: new list of weights, length 3 if second == False else 2
        """
        if second_derivative:
            return [
                2 * (w[1] - w[0]),
                2 * (w[2] - [1]),
            ]
        elif not second_derivative:
            return [
                3 * (w[1] - w[0]),
                3 * (w[2] - [1]),
                3 * (w[3] - w[2])
            ]
    
    def first_derivative_cubic(self, t: float, w: list[float]) -> float:
        w = self._calculate_derivative_weights(w)
        return self._quadratic_bezier(t, w)
    
    def second_derivative_cubic(self, t: float, w: list[float]) -> float:
        w = self._calculate_derivative_weights(w, second_derivative=True)
        return self._n_bezier_(1, t, w)
    
    def get_cubic_derivative(self, wx: list[float], wy: list[float], t: float, second_derivative: bool = False) -> list[float]:
        if not second_derivative:
            return [
                self.first_derivative_cubic(t, wx),
                self.first_derivative_cubic(t, wy)
            ]
        elif second_derivative:
            return [
                self.second_derivative_cubic(t, wx),
                self.second_derivative_cubic(t, wy)
            ]
        
                 
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
    