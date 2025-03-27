from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from track_gen.abstract import abstract_track
    import numpy as np

class TrackGenerator():

    def generate_track(self, seed: int, config: dict) -> abstract_track.Track:
        """
        A method that given a seed, generates a track.
        
        Args: 
            seed: int
                A integer from 0 to 43918403
        """
        pass   
    
    def calculate_genome(self, x: int, y:int) ->  np.ndarray:
        """
            Takes in an x and y value, returns the following genotype
            ```python
                [x, y, slope, c1.x, c1.y, c2.x, c2.y]
            ````
        """
        return
    
    def fitness(self, track: abstract_track.Track, config: dict) -> float:
        """
            Calculates the fitness of the track provided
        """
        return 
    
    def mutate(self, track: abstract_track.Track, config: dict) -> abstract_track.Track:
        """
            Picks a random index and moves the control point within the 
            bounds defined by the previous and next point, keeping the
            shape uniform and not breaking it up
        """
    
    def crossover(self, parents: list[abstract_track.Track]) -> list[abstract_track.Track]:
        """
            Performs crossover on pairs of parents 
        """   