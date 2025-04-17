from __future__ import annotations
from track_gen.abstract import abstract_track_generator
from track_gen import utils
from track_gen.tracks import convex_hull_track
from typing import TYPE_CHECKING

from concave_hull import concave_hull_indexes

if TYPE_CHECKING:
    from track_gen.abstract import abstract_track


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
        
        # for every point
        # calculate all closests points
        # if any < threshold then generate new point
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
        offsets = rng.uniform(-0.1, 0.1, (_num_points))
        perp_slopes = perp_slopes + offsets    
                
        # calculate the y intercepts
        y_intercepts = utils.LinearAlgebra.get_y_intercept(perp_slopes, hull_points[:, 1], hull_points[:, 0])
        
        c1 = np.ndarray(shape=(_num_points, 2))
        c2 = np.ndarray(shape=(_num_points, 2))
           
        # another constraint, control points MUST NOT overlap
        for point in range(len(hull_points)):
            current_point = hull_points[point]
            
            control_points = utils.LinearAlgebra.linear_eq(
                perp_slopes[point], current_point[0], y_intercepts[point], -config["weight_point_offset"], config["weight_point_offset"], 2
            )
            
            # get the first control point, 
            c1[point] = control_points[0]

            # get the second control point, 
            c2[point] = control_points[1]
        
        
        # encode the points
        track.encode_control_points(
            hull_points[:, 0, np.newaxis], hull_points[:, 1, np.newaxis], 
            perp_slopes[:, np.newaxis], 
            c1[:, 0, np.newaxis], c1[:, 1, np.newaxis], 
            c2[:, 0, np.newaxis], c2[:, 1, np.newaxis]
        )
            
        # calculate bezier
        track.calculate_bezier(config)
        return track
    
    def calculate_genome(self, x: float, y: float, seed: int, config: dict) -> np.ndarray:
        rng = np.random.default_rng(seed=seed)
        
        point = np.asarray([[x,y]])
        slopes = utils.LinearAlgebra.calculate_slopes(point)    
    
        # calculate the perpendicular slope and its gradient
        perp_slopes = utils.LinearAlgebra.calculate_slope_tangent(slopes)
        
        # apply some random additions / subractions to the slope, breaks up the shape
        offsets = rng.uniform(-0.1, 0.1, (1))
        perp_slopes = perp_slopes + offsets    
                
        # calculate the y intercepts
        y_intercepts = utils.LinearAlgebra.get_y_intercept(perp_slopes, point[:, 1], point[:, 0])
         
        control_points = utils.LinearAlgebra.linear_eq(
            perp_slopes[0], point[0][0], y_intercepts[0], -config["weight_point_offset"], config["weight_point_offset"], 2
        )
        
        # get the first control point, 
        c1 = control_points[0]
        # get the second control point, 
        c2 = control_points[1]

        return np.asarray([
            point[0][0], point[0][1], perp_slopes[0], c1[0], c1[1], c2[0], c2[1]
        ])
    
    def mutate(self, track: abstract_track.Track, config: dict) -> abstract_track.Track:
        """
            Picks a random index and moves the point within the 
            bounds defined by the previous and next point, keeping the
            shape uniform and not breaking it up
        """
            
        rng = np.random.default_rng(seed=track.seed)
        
        curr_index = rng.integers(0, config["control_points"], size=1)
        next_index = utils.clamp(curr_index + 1, 0, config["control_points"])
        prev_index = utils.clamp(curr_index - 1, 0, config["control_points"])
        
        track_genotype = track.get_genotype()

        x_range = track_genotype[next_index][0][0] >= track_genotype[prev_index][0][0]
        x_bounds = [
            track_genotype[prev_index][0][0] if x_range else track_genotype[next_index][0][0],
            track_genotype[prev_index][0][0] if not x_range else track_genotype[next_index][0][0]    
        ]
        
        y_range = track_genotype[next_index][0][1] >= track_genotype[prev_index][0][1]
        y_bounds = [
            track_genotype[prev_index][0][1] if y_range else track_genotype[next_index][0][1],
            track_genotype[prev_index][0][1] if not y_range else track_genotype[next_index][0][1]    
        ]

        # generate coordinate 
        x_coords = rng.uniform(x_bounds[0], x_bounds[1], 1)
        y_coords = rng.uniform(y_bounds[0], y_bounds[1], 1)
            
        mutated_control_point = self.calculate_genome(x_coords[0], y_coords[0], track.seed, config)
        track.encode_control_point(curr_index, *mutated_control_point)
        
        # calculate bezier
        track.calculate_bezier(config)
        
        return track
         
    def crossover(self, parents: list[abstract_track.Track], config: dict) -> list[abstract_track.Track]:
        """
            Performs crossover on pairs of parents 
        """        
        
        offspring = []
        _parents = len(parents)
        
        # instead of getting crossover point as an index arrary
        # cross on a random line through origin
        
        
        # edge condition where list of parents does not contain enough parents
        if not _parents % 2 == 0:
            return parents
                
        for i in range(0, _parents, 2):
            # use the seed of the first parent
            p1 = parents[i]
            p2 = parents[i+1]
           
            p1_geno = p1.get_genotype()
            p2_geno = p2.get_genotype()
                       
            rng = np.random.default_rng(seed=p1.seed)

            # generate random angle
            crossover_slope = np.sin(rng.integers(low=0, high=360, size=(1)))
            
            # for every point in both genotypes
            # get above and below the line
            # only crossover above / below
            _num_cp = p1._control_points
            
            # split parents into two arrays
            pos_delta__p1 = []
            pos_delta__p2 = []
            
            for index in range(_num_cp):
                # parent one
                pgenotypes = [
                    p1_geno[index],
                    p2_geno[index]
                ]
                
                p_ly = [
                    utils.LinearAlgebra.line_eq(crossover_slope, pgenotypes[0][0]),
                    utils.LinearAlgebra.line_eq(crossover_slope, pgenotypes[1][0])
                ]
                
                pos_delta__p1.append(False if p_ly[0] > pgenotypes[0][1] else True)
                pos_delta__p2.append(False if p_ly[1] > pgenotypes[1][1] else True)
                    
            pos_delta__p1 = np.asarray(pos_delta__p1)
            pos_delta__p2 = np.asarray(pos_delta__p2)
            
            p1_crossover = np.where((pos_delta__p1 == pos_delta__p2)[:, np.newaxis], p1_geno, p2_geno).T
            p2_crossover = np.where((pos_delta__p1 == pos_delta__p2)[:, np.newaxis], p2_geno, p1_geno).T     

            offspring.append(convex_hull_track.ConvexHullTrack(p1._control_points, p1.seed))
            offspring.append(convex_hull_track.ConvexHullTrack(p2._control_points, p2.seed))
                        
            # encode control points
            # attempted to unpack array using *
            # unfortunately results in the wrong shape of
            # (1, 10) instead of (10,1)
            offspring[i].encode_control_points(
                p1_crossover[0].T[:, np.newaxis],
                p1_crossover[1].T[:, np.newaxis],
                p1_crossover[2].T[:, np.newaxis],
                p1_crossover[3].T[:, np.newaxis],
                p1_crossover[4].T[:, np.newaxis],
                p1_crossover[5].T[:, np.newaxis],
                p1_crossover[6].T[:, np.newaxis]
            )
            offspring[i+1].encode_control_points(
                p2_crossover[0].T[:, np.newaxis],
                p2_crossover[1].T[:, np.newaxis],
                p2_crossover[2].T[:, np.newaxis],
                p2_crossover[3].T[:, np.newaxis],
                p2_crossover[4].T[:, np.newaxis],
                p2_crossover[5].T[:, np.newaxis],
                p2_crossover[6].T[:, np.newaxis]
            )
                        
            # calculate bezier for each
            offspring[i].calculate_bezier(config)   
            offspring[i+1].calculate_bezier(config)   
        
        return offspring
            
            
            
            
            
            

        