from __future__ import annotations
from typing import TYPE_CHECKING 

# Module imports
from track_gen.abstract import abstract_track_generator
from track_gen.tracks import convex_hull_track
from track_gen.utils import LinearAlgebra
from track_gen import utils
import track_gen.bezier as bezier


# Library imports
import concave_hull as ch
import numpy as np

# Type imports
if TYPE_CHECKING:
    from track_gen.abstract import abstract_track
    

class TrackGenerator(abstract_track_generator.TrackGenerator):
    
    def __init__(self, config: dict) -> None:
        self._bezier = bezier.Bezier(1, 0.01)
        self.config = config
    
    def generate_track(self, seed: int):
        self.seed = np.random.default_rng(seed)
        _num_points = self.config['control_points']
                
        # new track object
        track = convex_hull_track.ConvexHullTrack(_num_points, seed)
        
        # generate new points and check if within threshold
        points = self._within_threshold(self._initialise_points(_num_points))

        # generate hull points
        hull = self._concave_hull(
            points, _num_points
        )     
        
        # calculate and encode the control points
        control = self._calculate_control_points(track, _num_points, hull)    
        bezier_segments = self._calculate_bezier(_num_points, control, hull) 
        
        # WHEN DOING CROSS OVER
        # RE CALCULATE
        # BEZIER SEGMENTS
        # CURV PROFILE
        # BEZIER_COORIDNATES (only for rendering)
    
        # encode the bezier segments
        track.BEZIER_SEGMENTS = np.asanyarray(bezier_segments)
        # get the curvature profile of the track
        track.CURVATURE_PROFILE = self._curvature_profile(track)
        
        # calculate full track coordinates
        track.BEZIER_COORDINATES = self._track_coordinates(track)
        
        # calculate track length
        track = self._track_length(track)
        return track
    
    def _track_length(self, track: abstract_track.Track) -> abstract_track.Track:
        track.LENGTH = np.sum(track.BEZIER_SEGMENTS[:, 8])       
        return track

    def _track_coordinates(self, track: abstract_track.Track) -> abstract_track.Track:
        # for each bezier segment
        coordinates = [] 
        
        for segment in track.BEZIER_SEGMENTS: 
            wx = segment[0:7:2] 
            wy = segment[1:8:2] 
            coordinates.extend(self._bezier.generate_bezier(self._bezier.CUBIC, wx, wy, 0.01))

        return coordinates
            
    def _initialise_points(self, num_points: int) -> np.ndarray:
        """
            Generates `num_points` random points with bounds defined in`config['x_bounds']`and`config['y_bounds']`

        """
        # based on the bounds from the config, generate control points
        x_bounds = self.config["x_bounds"]
        y_bounds = self.config["y_bounds"]
        
        # generate coordinates 
        x_coords = self.seed.uniform(x_bounds["low"], x_bounds["high"], num_points)[:, np.newaxis]
        y_coords = self.seed.uniform(y_bounds["low"], y_bounds["high"], num_points)[:, np.newaxis]
        
        return np.column_stack((x_coords, y_coords))
        
    def _within_threshold(self, points: np.ndarray) -> np.ndarray:
        """
            Calculate the distance for all neighbouring points. If the distance is below the threshold, push the 
            points apart.
        """
        threshold_dist = self.config['threshold_distance']
    
        for point in range(points.shape[0]):
            for point_next in range(points.shape[0]):
                
                if point == point_next:
                    continue
                
                # distance 
                distance = np.linalg.norm(points[point] - points[point_next])
                
                if distance < threshold_dist:
                    # calculate the vector from point[point] to points[point_next]
                    # normalise by magnitude
                    norm_vec = (points[point] - points[point_next]) / distance 
                    distance_offset = (threshold_dist - distance) * norm_vec
                    
                    # offset point                   
                    points[point] += distance_offset
        return points        
            
    
    def _concave_hull(self, points: np.ndarray, num_points: int) -> np.ndarray:
        """
            Calculates the concave hull of the given points. If the number of points within the concave hull is less than `num_points`,
            regenerate missing points.
        """
        
        concave_hull = ch.concave_hull_indexes(points, concavity=0, length_threshold=0)
        hull_points = points[concave_hull]
        
        x_bounds = self.config['x_bounds']
        y_bounds = self.config['y_bounds']
        
        while (hull_points.shape[0] < num_points):
            # get all the indexes of points that arent part of concave hull
            # generate new points till concave hull covers all            
            bad_points_mask = ~(np.all(points[:, None] == hull_points, axis=-1).any(axis=1))
            
            index = 0 
            for bad_point in bad_points_mask:
                if bad_point:
                    points[index] = np.column_stack((
                        self.seed.uniform(x_bounds["low"], x_bounds["high"], 1)[:, np.newaxis],
                        self.seed.uniform(y_bounds["low"], y_bounds["high"], 1)[:, np.newaxis]
                    ))
                index += 1    
            
            # make sure points are above threshold
            points = self._within_threshold(points)    
            # calculate new concave hull  # get the concave hull
            concave_idx = ch.concave_hull_indexes(points, concavity=0, length_threshold=0)
            hull_points = points[concave_idx]     
                  
        return hull_points 
        
    def _calculate_control_points(self, track: convex_hull_track.abstract_track, num_points: int, points: np.ndarray) -> np.ndarray:
        """
            Calculates the control points used in the bezier curve.
        """
        slopes = utils.LinearAlgebra.calculate_slopes(points)
        perp_slopes = utils.LinearAlgebra.calculate_slope_tangent(slopes)
        
        # apply a random offset to the gradient 
        # this breaks up the shape of the track, makes it a bit more natural
        slope_offset = self.seed.uniform(self.config['slope_offset']['low'], self.config['slope_offset']['high'], (num_points))
        perp_slopes = perp_slopes + slope_offset
        
        # calculate the y intercepts
        y_intercepts = utils.LinearAlgebra.get_y_intercept(perp_slopes, points[:, 1], points[:, 0])
        
        # initialise arrays for control points
        c1 = np.ndarray(shape=(num_points, 2))
        c2 = np.ndarray(shape=(num_points, 2))
        
        # using the function utils.LinearAlgebra.linear_eq
        # calculate two points along the line of perp_slopes
        # with an equal offset defined in config["control_point_offset"]

        control_offset = self.config['control_point_offset'] 

        for point in range(num_points):
            current_point = points[point]
            
            _control = utils.LinearAlgebra.linear_eq(
                perp_slopes[point], current_point[0], y_intercepts[point], -control_offset, control_offset, 2
            )
            c1[point] = _control[0]
            c2[point] = _control[1]
            
        # encode the control points
        track.encode_control_points(
            points[:, 0, np.newaxis],
            points[:, 1, np.newaxis],
            slopes[:, np.newaxis],
            c1[:, 0, np.newaxis],
            c1[:, 1, np.newaxis],
            c2[:, 0, np.newaxis],
            c2[:, 1, np.newaxis]
        )

        return np.hstack((c1, c2))
    
    def _calculate_bezier(self, num_points: int, control_points: np.ndarray, points: np.ndarray) -> np.ndarray:
        """
            Calculates the weightings used for bezier curves
            Encodes the resulting bezier curves as segments.
            
        """
        reserved_control = []
        segments = []
                
        for idx in range(num_points):
            n_idx = utils.clamp(idx + 1, 0, num_points)            
            
            segment = np.zeros(shape=(1,0))
                    
            # calculate distances between the control points and track points
            dist_p1_n1 = LinearAlgebra.euclidean_distance(points[idx], control_points[n_idx, 0:2])
            dist_p1_n2 = LinearAlgebra.euclidean_distance(points[idx], control_points[n_idx, 2:4])
            
            dist_p2_c1 = LinearAlgebra.euclidean_distance(points[n_idx], control_points[idx, 0:2])
            dist_p2_c2 = LinearAlgebra.euclidean_distance(points[n_idx], control_points[idx, 2:4])

            # choose the closer points as weightings
            c1 = control_points[idx, 0:2] if dist_p2_c1 < dist_p2_c2 else control_points[idx, 2:4]
            c2 = control_points[n_idx, 0:2] if dist_p1_n1 < dist_p1_n2 else control_points[n_idx, 2:4]
            
            # check control points are reserverd
            # this stops point like corners forming
            # if point is reserved, swap over
            if c1 in reserved_control: c1 = control_points[idx, 0:2] if dist_p2_c1 > dist_p2_c2 else control_points[idx, 2:4]
            if c2 in reserved_control: c2 = control_points[n_idx, 0:2] if dist_p1_n1 > dist_p1_n2 else control_points[n_idx, 2:3]
            
            weights = np.vstack((
                points[idx], c1, c2, points[n_idx]
            ))
                
            # define start, w1, w2, end
            segment = np.insert(segment, 0, np.hstack((points[idx], c1, c2, points[n_idx])))
                                                             
            # use a rough approximation of the bezier curve to calculate the arc length
            length = self._bezier.approx_arc_length(weights[:, 0], weights[:, 1])
    
            # add length to segment
            segment = np.insert(segment, 8, length)
            
            segments.append(segment)
        return segments        

    def _curvature_profile(self, track: abstract_track.Track) -> abstract_track.Track: 
        segment_curv = []
        for segment in track.BEZIER_SEGMENTS:
            wx = segment[0:7:2] 
            wy = segment[1:8:2] 
            length = segment[8] 
            
            # calculate t intervals for a fixed distance of 1
            t = self._bezier.fixed_distance_interval(wx, wy, length)
                 
            # get curvature of the curve
            segment_curv.append(self._bezier.get_bezier_curvature_t(wx, wy, t))
        return segment_curv
    