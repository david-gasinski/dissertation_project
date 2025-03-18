from __future__ import annotations
from track_gen.abstract import abstract_track_generator
from track_gen.tracks import convex_hull_track
from typing import TYPE_CHECKING

import numpy as np

class ConvexHullGenerator(abstract_track_generator.TrackGenerator):
    
    def generate_track(self, seed: int, config: dict) -> convex_hull_track.ConvexHullTrack:
        # get the number of control points from config
        _num_points = config["control_points"]
        
        # initialise a track object
        track = convex_hull_track.ConvexHullTrack(_num_points)

        # init the rng generator
        rng = np.random.default_rng(seed=seed)
        
        # based on the bounds from the config, generate control points
        x_bounds = config["x_bounds"]
        y_bounds = config["y_bounds"]
        
        # generate coordinates 
        x_coords = rng.uniform(x_bounds["low"], x_bounds["high"], _num_points)
        y_coords = rng.uniform(y_bounds["low"], y_bounds["high"], _num_points)
        
        points = np.column_stack((x_coords, y_coords))
        
        # sort the array by x 
        #points = points[points[:, 0].argsort()]
        
        # encode the points
        
        # for the time being, use these values for slope
        test_val = np.ones(shape=(_num_points, 1), dtype=np.float32) * 20.3
        
        
        return super().generate_track(seed)