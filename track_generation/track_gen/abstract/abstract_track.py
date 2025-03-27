from __future__ import annotations
from typing import TYPE_CHECKING
import json

import numpy as np
if TYPE_CHECKING:
    import pygame

class Track():
    
    CONTROL_POINTS = None
    BEZIER_COORDINATES = None # calculate
    
    def __init__(self, control_points: int, seed: int) -> None:
        """
            Initialise the control point numpy array 
        
            Args:
                control_points: int
                Number of control points in the track

        """
        self._control_points = control_points
        self.seed = seed
        self.CONTROL_POINTS = np.ndarray(shape=(control_points, 7), dtype=np.float32)
    
    def encode_control_point(self, index: int, x: float, y: float, slope: float, x_c1: float, y_c1: float, x_c2: float, y_c2: float) -> None:
        """
            Encodes a control point as a numpy array to the object
            
            Args
                x: float
                y: float
                slope: float
                x_c1: float
                y_c1: float
                x_c2: float
                y_c2: float 
        """
        self.CONTROL_POINTS[index] = np.asanyarray([
            x,y,slope,x_c1,y_c1, x_c2, y_c2
        ], dtype=np.float32)
    
    
    def encode_control_points(self, x: np.ndarray, y: np.ndarray, slope: np.ndarray, x_c1: np.ndarray,  y_c1: np.ndarray, x_c2: np.ndarray, y_c2: np.ndarray) -> None:
        """
            Encodes a array of control points to the object
            
            Args:
                x: float
                y: float
                slope: float
                x_c1: float
                y_c1: float
                x_c2: float
                y_c2: float 
        """
        self.CONTROL_POINTS = np.hstack((x, y, slope, x_c1, y_c1, x_c2, y_c2))
        
    def render(self, screen: pygame.Surface) -> None:
        """
        A method to render the track 
        
        Args:
            screen: pygame.Surface
                Surface object to blit the track object on 
        
        Returns:
            None
        """
        pass
    
    def get_genotype(self) -> np.ndarray:    
        """
            Returns an encoded form of the genotype       
        """
        return self.CONTROL_POINTS

    def calculate_bezier(self, config: dict):
        """
            Using the genotype, calculates the bezier curves
        """
        return
    
    def fitness(self) -> float:
        """
            Calculates the fitness based on the following heuristics
            
            Starting fitness = 100
                - -5 per intersection on track
        """
        return 

    def convert_to_screen(self, coords):
        """
            Used for rendering, converts to screen coordinates
        """
        return coords + 450
