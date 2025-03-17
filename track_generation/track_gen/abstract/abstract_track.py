from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pygame
    import numpy as np

class Track():
    
    CONTROL_POINTS = []
    
    
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
    
    def fitness(self) -> float:
        """
            Calculates the fitness of the track 
        """
        return
    
    def mutate(self, mutation_points: int = 1) -> None:
        """
            Randomly mutate a control point on the track
        """
        
        
    def get_genotype(self) -> np.ndarray:    
        """
            Returns an encoded form of the genotype
            
        """
        return
