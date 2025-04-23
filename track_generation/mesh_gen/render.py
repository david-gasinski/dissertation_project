from track_gen.abstract import abstract_track
from track_gen.generators import track_generator

from track_gen import utils

class TrackMesh():
    TMP = "mesh_gen\\temp"
    
    def __init__(self, track: abstract_track.Track, config: dict):
        self.track = track
        self.config = config
        self.track_gen = track_generator.TrackGenerator(config)
        
    def serialize_track(self) -> None:
        serialized_track = self.track.serialize()
        
        # using the genotype generate a new bezier curve using a smaller interval
        self.track_gen._track_coordinates(
            track=self.track, interval=0.01
        )
        serialized_track['track_coords'] = self.track.TRACK_COORDS.tolist()
        
        return serialized_track
    
    def save_track(self) -> None:
        raw_track = self.serialize_track()
        