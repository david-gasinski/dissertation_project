from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from track_gen.abstract import abstract_track

class TrackGenerator():

    def generate_track(self, seed: int) -> abstract_track.Track:
        """
        A method that given a seed, generates a track.
        
        Args: 
            seed: int
                A integer from 0 to 43918403
        """
        pass
    
# generate track -> return Track object as genotype