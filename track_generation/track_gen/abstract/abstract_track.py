from __future__ import annotations
from typing import TYPE_CHECKING
import json
import codecs

import numpy as np
if TYPE_CHECKING:
    import pygame

class Track():
    
    CONTROL_POINTS = None
    BEZIER_SEGMENTS = None # calculate
    CURVATURE_PROFILE = None # calculate
    TRACK_COORDS = None
    LENGTH = None
    
    def __init__(self, control_points: int, seed: int) -> None:
        """
            Initialise the control point numpy array 
        
            Args:
                control_points: int
                Number of control points in the track

        """
        self._control_points = control_points
        self.seed = seed
        self._fitness = 100 # default fitness
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
        
    def encode_bezier_segments(self, segments: np.ndarray) -> None:
        """
            Encodes a list of segments onto an track object
            
            Each segment is an array containing  
                start: [float, float]
                w1: [float, float]
                w2: [float, float]
                end: [float, float]
                length: float
            
            Args:
                segments: list
        """
        self.BEZIER_SEGMENTS = segments
        
    def encode_curvature_profile(self, curv: list) -> None:
        """
            Encodes a curvature profile onto an track object

            Args:
                curv: list
        """
        self.CURVATURE_PROFILE = curv
        
    def encode_track_coordinates(self, track_coords: np.ndarray) -> None:
        """
            Encodes track coordinates onto an track object

            Args:
                track_coords: list
        """
        self.TRACK_COORDS = track_coords
        
    def encode_upper_offset(self, track_coords: np.ndarray) -> None:
        """
            Encodes track coordinates onto an track object

            Args:
                track_coords: list
        """
        self.UPPER_TRACK_COORDS = track_coords
        
    def encode_track_length(self, length: float) -> None:
        """
            Encodes length of a track onto a track object
        """
        self.LENGTH = length
        
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
        return self._fitness
    
    def encode_fitness(self, fitness: float) -> None:
        self._fitness = fitness


    def convert_to_screen(self, coords):
        """
            Used for rendering, converts to screen coordinates
        """
        return coords + 450


    def serialize(self) -> str:
        """
            Serializes track, including
                self.CONTROL_POINTS 
                self.seed
                self.fitness    
            to be saved in json file
        """
        return json.dumps({
            "seed": self.seed,
            "fitness" : self._fitness,
            "genotype": self.CONTROL_POINTS.tolist()
        })
        
    