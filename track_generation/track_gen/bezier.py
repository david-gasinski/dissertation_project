from __future__ import annotations

import numpy as np
import scipy
import scipy.datasets
import scipy.spatial
from track_gen import utils

from typing import Union

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
        )) # remove last index as its the distance from 0 -> -1
        
        # remove last index as its the distance from 0 -> -1
        # but only if the shape doesnt match
        if (dist.shape != distance):
            dist = dist[:-1]
    
        # get the cumulative sum of distances, x axis for interpolation
        cum_dist = np.cumsum(dist)
                
        # associated t values used to generate the curve
        t = np.linspace(0, 1, int(distance)) # y value for interpolation
        
        # generate an array of size distance with equal intervals 
        # int() rounds down, so this works
        # if int() rounds up, you would be extrapolating (not interpolating) each curve
        interp_dist = np.linspace(0, int(distance) - 1, int(distance)) 

        # interpolate values of t based on interp_dist and cum_dist
        # essentially, plot t vs cum_dist
        # interpolate each value of interp_dist, find t
        # this is the t value needed to generate a curvature profile
                
        return np.interp(interp_dist, cum_dist, t)
    
    def closest_point(self, p: list[float], curve: list[float]) -> list[float]:
        """
            Returns the coordinates of the closest point along the curve
        """
        curve = np.asanyarray(curve)

        # calculate distance point to curve
        dist = scipy.spatial.distance.cdist(
            curve, [p], 'euclidean'
        )        
        # get the index of the smallest dist value
        closest_idx = np.argmin(dist)
                
        # return the closest point
        return curve[closest_idx].tolist()
        

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
       
    def find_roots(self, wx: list[float], wy: list[float]) -> list[float]:
        # get first derivate weights
        w = self._calculate_derivative_weights(wx)
        
        # calculate a b c
        ax = w[0] - (2 * w[1] + w[2])
        bx = 2 * (w[1] - w[0])
        cx = w[0] 
        
        # quadratic formula to find value of t when B(t) = 0
        t_roots = utils.PolynomialAlgebra.quadratic_formula(float(ax), float(bx), float(cx))
        
        roots = []
        # for each root
        # calculate the value of B(t) for each weight
        for t in t_roots:
            x = self._cubic_bezier(t, wx)
            y = self._cubic_bezier(t, wy)
            
            roots.append([x,y]) 
    
        return roots
    
    def offset_curve(self, offset_dist: float, curve: Union[list[float], np.ndarray], wx: list[float], wy: list[float]) -> list[float]:
        # find the roots of the curve
        roots = self.find_roots(wx, wy)
        
        flip_sign = False
        if offset_dist < 0:
            flip_sign = True
            offset_dist = abs(offset_dist)
        
        # first the closest index to each root
        # calculate the points for each index
        p_roots = []
        for root in roots:
            p_roots.append(curve.index(self.closest_point(root, curve)))
            
        # check if roots even exist
        # if they dont, partion three ways way
        if not p_roots[0] and not p_roots[1]:
            dist = len(curve)
            p_roots = [
                int(dist/3),
                int((dist/3) * 2)
            ]
        
        # sort to preserve order
        p_roots.sort()
        
        single_root = False
        # if curve only has one root (value is the same, or one if zero)
        if (p_roots[0] == p_roots[1]) or (not p_roots[0] or not p_roots[1]): 
            single_root = True
            
        # if the root of the parameteric function is non existent (aka the curve only has one local minima)
        # then the returned index is zero
        # in which case
        # we ignore it
        # and use the other     
                    
        # split the curve
        if single_root:
            segments = [
                curve[0:p_roots[1]] + utils.LinearAlgebra.offset_xy(offset_dist, flip_sign),
                curve[p_roots[1]:] + utils.LinearAlgebra.offset_xy(offset_dist, flip_sign)
            ]
        else:
            segments = [
                curve[0: p_roots[0]]  + utils.LinearAlgebra.offset_xy(offset_dist, flip_sign),
                curve[p_roots[0]: p_roots[1]] + utils.LinearAlgebra.offset_xy(offset_dist, flip_sign),
                curve[p_roots[1]:] + utils.LinearAlgebra.offset_xy(offset_dist, flip_sign)
            ]
                
        offset_curve = []
        # re calculate (for each segment) a new curve
        # use the same control points
        # use a smaller t offset for better performance


        # issue
        # returns this in the wrong format array
        # instead of 2d array shape (x,2)
        # returns array of shape x*2
        for segment in segments:
            curve_segment = self.generate_cubic_bezier_t(
                    [segment[0][0], wx[1], wx[2], segment[-1][0]],
                    [segment[0][1], wy[1], wy[2], segment[-1][1]],
                    0.1
                )
            #print(curve_segment)
            offset_curve.extend(
                curve_segment
            )
        
        return offset_curve
    
    
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
    
    def get_bezier_curvature_t(self, wx: list[float], wy: list[float], t: np.ndarray) -> list: 
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
    