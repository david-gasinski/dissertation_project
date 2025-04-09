from __future__ import annotations

import numpy as np
import scipy
import scipy.datasets
import scipy.spatial
from track_gen import utils

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

    def arc_length(self, points: list[float]) -> None:
        """
            Returns an approximation of the arc length
        """
        distance = 0
        num_points = len(points)

        for point in range(num_points - 1):
            p_current = points[point]
            p_next = points[utils.clamp(point + 1, 0, num_points)]
            
            # get linear distance
            distance += utils.LinearAlgebra.euclidean_distance(p_current, p_next)
        return distance
    
    def generate_cubic_bezier_t(self, wx: list[float], wy: list[float], t: np.ndarray) -> list[float]:
        mt = 1 - t
        
        bx = ((wx[0] * mt**3) + (wx[1] * 3 * mt** 2 * t) + (wx[2] * 3 * mt*t**2) + (wx[3] *t**3))
        by = ((wy[0] * mt**3) + (wy[1] * 3 * mt** 2 * t) + (wy[2] * 3 * mt*t**2) + (wy[3] *t**3))
        return np.hstack((bx, by)).tolist() # to list because i wrote everything without using numpy so cooked
    
    def fixed_distance_interval(self, wx: list[float], wy: list[float], distance: float) -> list[float]:
        """
            Returns t values of points along a parametric curve (defined by wx wy) at fixed distance intervals
            Based on the theory described in https://pomax.github.io/bezierinfo/#tracing
        """
        
        # firstly we need to calculate t interval
        t_interval = 1 / distance
        
        # np.ndarray of points curve points
        curve = np.asanyarray(self.generate_bezier(self.CUBIC, wx, wy, t_interval)) 
        
        # create a copy of the array but swap the first and last index
        # this ensures, when doing distance calculations
        # we calculate the distance from 0 -> 1, 1 -> 2, etc... (distance between successive points)
        # rather than as pairs, or 0 -> 0
        swap_curve = np.vstack((curve[1:, :],curve[0,:]))
        
        # calculate a distance matrix for curve and swap_curve
        dist = np.diag(scipy.spatial.distance.cdist(
            curve, swap_curve, 'euclidean'
        ))[:-1] # remove last index as its the distance from 0 -> -1
    
        # get the cumulative sum of distances, x axis for interpolation
        cum_dist = np.cumsum(dist)
                
        # associated t values used to generate the curve
        t = np.linspace(0, 1, int(distance)) # y value for interpolation
        
        # generate an array of size distance with equal intervals 
        # int() rounds down, so this works
        # if int() rounds up, you would be extrapolating (not interpolating) each curve
        interp_dist = np.linspace(0, int(distance), int(distance) + 1) 

        # interpolate values of t based on interp_dist and cum_dist
        # essentially, plot t vs cum_dist
        # interpolate each value of interp_dist, find t
        # this is the t value needed to generate a curvature profile
        
        return np.interp(interp_dist, cum_dist, t)

    def approx_arc_length(self, wx: list[float], wy: list[float]) -> float:
        """
            Given wx and wy, approximate the length of a bezier curve by flattening the curve and adding up the length
            of individual segments.
        """
        bezier_coords = self.generate_bezier(self.CUBIC, wx, wy, interval=0.1)
        return self.arc_length(bezier_coords)
    
    def _calculate_derivative_weights(self, w: list[float], second_derivative: bool = False) -> list[float]:
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
    
    def get_kappa(self, wx: list[float], wy: list[float], t: float):
        """
            Implementation based on the formula
            https://pomax.github.io/bezierinfo/#curvature
            
        """
        
        d = self.get_cubic_derivative(wx, wy, t)
        dd = self.get_cubic_derivative(wx, wy, t, second_derivative=True) 
        
        numerator = (d[0]*dd[1]) - (dd[0] * d[1])
        denominator = pow((d[0]*d[0]) + (d[1]*d[1]), 3/2)
            
        kappa = numerator / denominator
    
        return kappa
        
    def get_bezier_curvature(self, wx: list[float], wy: list[float], t_interval: float = None):
        """
            Returns the curvature of a bezier curve with weights (wx, wy) and 1 / t_interval points
        """
        curvature = []
        
        interval = self.interval if not t_interval else t_interval
        
        t = 0
        while t <= 1:
            curvature.append(self.get_kappa(wx, wy, t))
            t += interval            
        return curvature
    
    def get_bezier_curvature_t(self, wx: list[float], wy: list[float], t: np.ndarray):
        """
            Returns the curvature of a bezier curve with weights (wx, wy) along t 
        """
        t_values = len(t)
        curvature = np.ndarray(shape=t_values)
                
        for i in range(t_values):
            curvature[i] = self.get_kappa(wx, wy, t[i])     
            
        return curvature.tolist()
        
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
              
    def generate_bezier(self, curve_type: int,  wx: list[float], wy: list[float], interval: float = None) -> list[list[float]]:
        curve_coordinates = []
        t = 0
                
        while t <= 1:
            curve_coordinates.append(self._n_bezier(curve_type, wx, wy, t))
            t+= self.interval if interval is None else interval
        return curve_coordinates
    