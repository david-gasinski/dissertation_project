from __future__ import annotations
from track_gen.abstract import abstract_track_generator
from track_gen import utils
from track_gen.tracks import convex_hull_track
from typing import TYPE_CHECKING

from concave_hull import concave_hull, concave_hull_indexes

import numpy as np

class ConvexHullGenerator(abstract_track_generator.TrackGenerator):
    
    def generate_track(self, seed: int, config: dict) -> convex_hull_track.ConvexHullTrack:
        # get the number of control points from config
        _num_points = config["control_points"]
        
        # initialise a track object
        track = convex_hull_track.ConvexHullTrack(_num_points, seed)

        # init the rng generator
        rng = np.random.default_rng(seed=seed)
        
        # based on the bounds from the config, generate control points
        x_bounds = config["x_bounds"]
        y_bounds = config["y_bounds"]
        
        # generate coordinates 
        x_coords = rng.uniform(x_bounds["low"], x_bounds["high"], _num_points)[:, np.newaxis]
        y_coords = rng.uniform(y_bounds["low"], y_bounds["high"], _num_points)[:, np.newaxis]
        
        points = np.column_stack((x_coords, y_coords))
        
        # test the distance to k nearest coordinates
        # if too close, generate new coorindate
        threshold_distance = config['threshold_distance']
        
        for point in range(points.shape[0]):
            for point_next in range(points.shape[0]):
                
                if point == point_next:
                    continue
                
                # distance 
                distance = np.linalg.norm(points[point] - points[point_next])
                
                if distance < threshold_distance:
                    # calculate the vector from point[point] to points[point_next]
                    # normalise by magnitude
                    norm_vec = (points[point] - points[point_next]) / distance 
                    distance_offset = (threshold_distance - distance) * norm_vec
                    
                    # offset point                   
                    points[point] += distance_offset
                
        
        # get the concave hull
        concave_idx = concave_hull_indexes(points, concavity=0, length_threshold=0)
        hull_points = points[concave_idx]
        
        while (hull_points.shape[0] < _num_points):
            # get all the indexes of points that arent part of concave hull
            # generate new points till concave hull covers all            
            bad_points_mask = ~(np.all(points[:, None] == hull_points, axis=-1).any(axis=1))
            
            index = 0 
            for bad_point in bad_points_mask:
                if bad_point:
                    points[index] = np.column_stack((
                        rng.uniform(x_bounds["low"], x_bounds["high"], 1)[:, np.newaxis],
                        rng.uniform(y_bounds["low"], y_bounds["high"], 1)[:, np.newaxis]
                    ))
                index += 1    
                                
            # calculate new concave hull  # get the concave hull
            concave_idx = concave_hull_indexes(points, concavity=0, length_threshold=0)
            hull_points = points[concave_idx]            

        # for each point, calculate its slope towards the origin
        slopes = utils.LinearAlgebra.calculate_slopes(hull_points)    
    
        # calculate the perpendicular slope and its gradient
        perp_slopes = utils.LinearAlgebra.calculate_slope_tangent(slopes)
        
        # apply some random additions / subractions to the slope, breaks up the shape
        offsets = rng.uniform(-0.1, 0.1, (10))
        perp_slopes = perp_slopes + offsets    
                
        # calculate the y intercepts
        y_intercepts = utils.LinearAlgebra.get_y_intercept(perp_slopes, hull_points[:, 1], hull_points[:, 0])
        
        c1 = np.ndarray(shape=(_num_points, 2))
        c2 = np.ndarray(shape=(_num_points, 2))
           
        # another constraint, control points MUST NOT overlap
        for point in range(len(hull_points - 1)):
            current_point = hull_points[point]
            
            control_points = utils.LinearAlgebra.linear_eq(
                perp_slopes[point], current_point[0], y_intercepts[point], -config["weight_point_offset"], config["weight_point_offset"], 2
            )
            
            # get the first control point, 
            c1[point] = control_points[0]

            # get the second control point, 
            c2[point] = control_points[1]
        
        
        #for i in range(config['straights']):
        #    index = rng.integers(0, 10, size=1)
        #    index_pair = utils.clamp(index + 1, 0, 10)
        #    
        #    c2[index] = [0,0]
        #    c1[index_pair] = [0,0]
        
        # encode the points
        track.encode_control_points(
            hull_points[:, 0, np.newaxis], hull_points[:, 1, np.newaxis], 
            perp_slopes[:, np.newaxis], 
            c1[:, 0, np.newaxis], c1[:, 1, np.newaxis], 
            c2[:, 0, np.newaxis], c2[:, 1, np.newaxis]
        )
        
        # calculate bezier
        track.calculate_bezier()
    
        return track