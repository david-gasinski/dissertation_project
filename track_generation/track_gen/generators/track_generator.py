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
import scipy.spatial as sp
import numpy as np

# Type imports
if TYPE_CHECKING:
    from track_gen.abstract import abstract_track


class TrackGenerator(abstract_track_generator.TrackGenerator):

    def __init__(self, config: dict) -> None:
        self._bezier = bezier.Bezier(1, 0.01)
        self.config = config

        # load the bins for entropy of curvature
        self.curvature_bins = utils.read_np(self.config["curvature_bin_dataset"])

    def generate_track(self, seed: int):
        self.seed = np.random.default_rng(seed)
        _num_points = self.config["control_points"]

        # new track object
        track = convex_hull_track.ConvexHullTrack(_num_points, seed)

        # generate new points and check if within threshold
        points = self._within_threshold(self._initialise_points())
        # generate hull points
        hull = self._concave_hull(points)

        # calculate and encode the control points
        self._calculate_control_points(track, hull)

        # calculate BEZIER_SEGMENTS, CURVATURE_PROFILE, TRACK_COORDS, LENGTH
        self._calc_track_params(track)

        return track

    def _track_length(self, track: abstract_track.Track) -> abstract_track.Track:
        track.encode_track_length(np.sum(track.BEZIER_SEGMENTS[:, 8]))

    def _track_coordinates(self, track: abstract_track.Track) -> abstract_track.Track:
        # for each bezier segment
        coordinates = []
        offset_coords = []
        offset = -70

        for segment in track.BEZIER_SEGMENTS:
            wx = segment[0:7:2]
            wy = segment[1:8:2]
            segment = self._bezier.generate_bezier(self._bezier.CUBIC, wx, wy, 0.1)
            coordinates.extend(segment)
            offset_coords.extend(self._bezier.offset_curve(offset, segment, wx, wy))

        track.encode_track_coordinates(np.asanyarray(coordinates))
        # track.encode_upper_offset(np.asanyarray(offset_coords)) change if using bezier offsets

    def _initialise_points(self) -> np.ndarray:
        """
        Generates `num_points` random points with bounds defined in`config['x_bounds']`and`config['y_bounds']`

        """
        # based on the bounds from the config, generate control points
        x_bounds = self.config["x_bounds"]
        y_bounds = self.config["y_bounds"]

        # generate coordinates
        x_coords = self.seed.uniform(
            x_bounds["low"], x_bounds["high"], self.config["control_points"]
        )[:, np.newaxis]
        y_coords = self.seed.uniform(
            y_bounds["low"], y_bounds["high"], self.config["control_points"]
        )[:, np.newaxis]

        return np.column_stack((x_coords, y_coords))

    def _within_threshold(self, points: np.ndarray) -> np.ndarray:
        """
        Calculate the distance for all neighbouring points. If the distance is below the threshold, push the
        points apart.
        """
        threshold_dist = self.config["threshold_distance"]

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

    def _concave_hull(self, points: np.ndarray) -> np.ndarray:
        """
        Calculates the concave hull of the given points. If the number of points within the concave hull is less than `num_points`,
        regenerate missing points.
        """

        concave_hull = ch.concave_hull_indexes(points, concavity=0, length_threshold=0)
        hull_points = points[concave_hull]

        x_bounds = self.config["x_bounds"]
        y_bounds = self.config["y_bounds"]

        while hull_points.shape[0] < self.config["control_points"]:
            # get all the indexes of points that arent part of concave hull
            # generate new points till concave hull covers all
            bad_points_mask = ~(
                np.all(points[:, None] == hull_points, axis=-1).any(axis=1)
            )

            index = 0
            for bad_point in bad_points_mask:
                if bad_point:
                    points[index] = np.column_stack(
                        (
                            self.seed.uniform(x_bounds["low"], x_bounds["high"], 1)[
                                :, np.newaxis
                            ],
                            self.seed.uniform(y_bounds["low"], y_bounds["high"], 1)[
                                :, np.newaxis
                            ],
                        )
                    )
                index += 1

            # make sure points are above threshold
            points = self._within_threshold(points)
            # calculate new concave hull  # get the concave hull
            concave_idx = ch.concave_hull_indexes(
                points, concavity=0, length_threshold=0
            )
            hull_points = points[concave_idx]

        return hull_points

    def _calculate_control_points(
        self, track: convex_hull_track.abstract_track, points: np.ndarray
    ) -> np.ndarray:
        """
        Calculates the control points used in the bezier curve.
        """
        slopes = utils.LinearAlgebra.calculate_slopes(points)
        perp_slopes = utils.LinearAlgebra.calculate_slope_tangent(slopes)

        # apply a random offset to the gradient
        # this breaks up the shape of the track, makes it a bit more natural
        slope_offset = self.seed.uniform(
            self.config["slope_offset"]["low"],
            self.config["slope_offset"]["high"],
            (self.config["control_points"]),
        )
        perp_slopes = perp_slopes + slope_offset

        # calculate the y intercepts
        y_intercepts = utils.LinearAlgebra.get_y_intercept(
            perp_slopes, points[:, 1], points[:, 0]
        )

        # initialise arrays for control points
        c1 = np.ndarray(shape=(self.config["control_points"], 2))
        c2 = np.ndarray(shape=(self.config["control_points"], 2))

        # using the function utils.LinearAlgebra.linear_eq
        # calculate two points along the line of perp_slopes
        # with an equal offset defined in config["control_point_offset"]

        control_offset = self.config["control_point_offset"]

        for point in range(self.config["control_points"]):
            current_point = points[point]

            _control = utils.LinearAlgebra.linear_eq(
                perp_slopes[point],
                current_point[0],
                y_intercepts[point],
                -control_offset,
                control_offset,
                2,
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
            c2[:, 1, np.newaxis],
        )

    def _calculate_bezier(self, track: abstract_track.Track) -> np.ndarray:
        """
        Calculates the weightings used for bezier curves
        Encodes the resulting bezier curves as segments.
        """
        reserved_control = []
        segments = []

        control_points = track.CONTROL_POINTS[:, 3:7]
        points = track.CONTROL_POINTS[:, 0:2]
        num_points = self.config["control_points"]

        for idx in range(num_points):
            n_idx = utils.clamp(idx + 1, 0, num_points)

            segment = np.zeros(shape=(1, 0))

            # calculate distances between the control points and track points
            dist_p1_n1 = LinearAlgebra.euclidean_distance(
                points[idx], control_points[n_idx, 0:2]
            )
            dist_p1_n2 = LinearAlgebra.euclidean_distance(
                points[idx], control_points[n_idx, 2:4]
            )

            dist_p2_c1 = LinearAlgebra.euclidean_distance(
                points[n_idx], control_points[idx, 0:2]
            )
            dist_p2_c2 = LinearAlgebra.euclidean_distance(
                points[n_idx], control_points[idx, 2:4]
            )

            # choose the closer points as weightings
            c1 = (
                control_points[idx, 0:2]
                if dist_p2_c1 < dist_p2_c2
                else control_points[idx, 2:4]
            )
            c2 = (
                control_points[n_idx, 0:2]
                if dist_p1_n1 < dist_p1_n2
                else control_points[n_idx, 2:4]
            )

            # check control points are reserverd
            # this stops point like corners forming
            # if point is reserved, swap over
            if c1.tolist() in reserved_control:
                c1 = (
                    control_points[idx, 0:2]
                    if dist_p2_c1 > dist_p2_c2
                    else control_points[idx, 2:4]
                )
            if c2.tolist() in reserved_control:
                c2 = (
                    control_points[n_idx, 0:2]
                    if dist_p1_n1 > dist_p1_n2
                    else control_points[n_idx, 2:4]
                )

            # add control points to reserved
            reserved_control.append(c1.tolist())
            reserved_control.append(c2.tolist())

            weights = np.vstack((points[idx], c1, c2, points[n_idx]))

            # define start, w1, w2, end
            segment = np.insert(
                segment, 0, np.hstack((points[idx], c1, c2, points[n_idx]))
            )

            # use a rough approximation of the bezier curve to calculate the arc length
            # round to the nearest integer
            length = round(self._bezier.approx_arc_length(weights[:, 0], weights[:, 1]))

            # add length to segment
            segment = np.insert(segment, 8, length)

            segments.append(segment)

        # assign segments to track
        track.encode_bezier_segments(np.asanyarray(segments))

    def _curvature_profile(self, track: abstract_track.Track) -> abstract_track.Track:
        segment_curv = (
            []
        )  # curvature has to be a list (not np.ndarray) as each segement has a different length
        for segment in track.BEZIER_SEGMENTS:
            wx = segment[0:7:2]
            wy = segment[1:8:2]
            length = segment[8]

            # calculate t intervals for a fixed distance of 1
            t = self._bezier.fixed_distance_interval(wx, wy, length)

            # get curvature of the curve
            curvature_profile = self._bezier.get_bezier_curvature_t(wx, wy, t)

            # ensure curvature_profile has the same size as segment length
            # this makes the profile loose some accuracy, but these are typically within
            # boundaries of curves, where the curvature is close to or equal to zero
            if len(curvature_profile) > length:
                diff = len(curvature_profile) - int(length)
                curvature_profile = curvature_profile[:-diff]

            segment_curv.extend(curvature_profile)  # append or extend
        track.encode_curvature_profile(segment_curv)

    def _calc_track_params(self, track: abstract_track.Track) -> None:
        """
        Calculates the following track parameters
            `self.BEZIER_SEGMENTS`
            `self.CURVATURE_PROFILE`
            `self.TRACK_COORDS`
            `self.LENGTH`
        """
        self._calculate_bezier(track)
        self._curvature_profile(track)
        self._track_length(track)

        self._track_coordinates(track)

    def crossover(
        self, parents: list[abstract_track.Track]
    ) -> list[abstract_track.Track]:
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
            p2 = parents[i + 1]

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
                pgenotypes = [p1_geno[index], p2_geno[index]]

                p_ly = [
                    utils.LinearAlgebra.line_eq(crossover_slope, pgenotypes[0][0]),
                    utils.LinearAlgebra.line_eq(crossover_slope, pgenotypes[1][0]),
                ]

                pos_delta__p1.append(False if p_ly[0] > pgenotypes[0][1] else True)
                pos_delta__p2.append(False if p_ly[1] > pgenotypes[1][1] else True)

            pos_delta__p1 = np.asarray(pos_delta__p1)
            pos_delta__p2 = np.asarray(pos_delta__p2)

            p1_crossover = np.where(
                (pos_delta__p1 == pos_delta__p2)[:, np.newaxis], p1_geno, p2_geno
            ).T
            p2_crossover = np.where(
                (pos_delta__p1 == pos_delta__p2)[:, np.newaxis], p2_geno, p1_geno
            ).T

            offspring.append(
                convex_hull_track.ConvexHullTrack(p1._control_points, p1.seed)
            )
            offspring.append(
                convex_hull_track.ConvexHullTrack(p2._control_points, p2.seed)
            )

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
                p1_crossover[6].T[:, np.newaxis],
            )
            offspring[i + 1].encode_control_points(
                p2_crossover[0].T[:, np.newaxis],
                p2_crossover[1].T[:, np.newaxis],
                p2_crossover[2].T[:, np.newaxis],
                p2_crossover[3].T[:, np.newaxis],
                p2_crossover[4].T[:, np.newaxis],
                p2_crossover[5].T[:, np.newaxis],
                p2_crossover[6].T[:, np.newaxis],
            )

            # re calculate segments, curvature profile and track coordinates for each track
            self._calc_track_params(offspring[i])
            self._calc_track_params(offspring[i + 1])
        return offspring

    def uniform_crossover(
        self, parents: list[abstract_track.Track]
    ) -> list[abstract_track.Track]:
        """
        Performs single-point crossover on pairs of parents
        """

        offspring = []
        _parents = len(parents)

        # edge condition where list of parents does not contain enough parents
        if not _parents % 2 == 0:
            return parents

        for idx in range(0, _parents, 2):
            p1 = parents[idx]
            p2 = parents[idx + 1]

            p1_geno = p1.get_genotype()
            p2_geno = p2.get_genotype()

            rng = np.random.default_rng(seed=p1.seed)

            # get the coordinates for each parent
            p1_xy = np.hstack((p1_geno[:, 0, np.newaxis], p1_geno[:, 1, np.newaxis]))
            p2_xy = np.hstack((p2_geno[:, 0, np.newaxis], p2_geno[:, 1, np.newaxis]))

            # generate distance matrix between point 1 and point 2
            dist = sp.distance.cdist(p1_xy, p2_xy, "euclidean")

            # for each offspring
            # loop from 1 to 10
            # select the first point from p1
            # select closest point from p2
            # pick between either

            for i in range(2):  # 2 offspring
                offspring.append(convex_hull_track.ConvexHullTrack(
                    p1._control_points, p1.seed if i == 1 else p2.seed
                )) # create a new track
                occupied_idx = []  # points which are already used
                temp_offspring = []

                for j in range(p1._control_points):
                    dist_vector = dist[j]
                    sorted_idx = dist_vector.argsort()
                    min_value = sorted_idx[0]  # sort to get closest points

                    x = 1
                    while (
                        min_value in occupied_idx
                    ):  # make sure non of the points are already used
                        if x == len(sorted_idx):
                            # if all occupied just use closest point
                            min_value = sorted_idx[0]
                            break
                        min_value = sorted_idx[i]

                    # randomly select a point for the offspring
                    prob = rng.random()

                    temp_offspring.append(
                        p1_geno[j] if prob > 0.5 else p2_geno[min_value]
                    )

                # convert to numpy array
                temp_offspring = np.asanyarray(temp_offspring)

                offspring[idx + i].encode_control_points(temp_offspring[:, 0, np.newaxis],
                    temp_offspring[:, 1, np.newaxis],
                    temp_offspring[:, 2, np.newaxis],
                    temp_offspring[:, 3, np.newaxis],
                    temp_offspring[:, 4, np.newaxis],
                    temp_offspring[:, 5, np.newaxis],
                    temp_offspring[:, 6, np.newaxis],
                )

                # recalculate other parameters and add offspring
                self._calc_track_params(offspring[idx + i])    
        return offspring

    def mutate(self, track: abstract_track.Track) -> abstract_track.Track:
        mutate_point = self.seed.integers(0, self.config["control_points"], size=1)

        p_idx = utils.clamp(mutate_point + 1, 0, self.config["control_points"])
        n_idx = utils.clamp(mutate_point - 1, 0, self.config["control_points"])

        track_geno = track.get_genotype()
        track_points = np.hstack(
            (track_geno[:, 0, np.newaxis], track_geno[:, 1, np.newaxis])
        )

        # calculate the x and y bounds
        x_range = track_points[n_idx][0][0] >= track_points[p_idx][0][0]
        x_bounds = [
            track_points[p_idx][0][0] if x_range else track_points[n_idx][0][0],
            track_points[p_idx][0][0] if not x_range else track_points[n_idx][0][0],
        ]

        y_range = np.any(track_points[n_idx][0][1] >= track_points[p_idx][0][1])

        y_bounds = [
            track_points[p_idx][0][1] if y_range else track_points[n_idx][0][1],
            track_points[p_idx][0][1] if not y_range else track_points[n_idx][0][1],
        ]

        # generate new coordinate
        x_coords = self.seed.uniform(x_bounds[0], x_bounds[1], 1)
        y_coords = self.seed.uniform(y_bounds[0], y_bounds[1], 1)

        # replace old point
        track_points[mutate_point] = np.asanyarray([x_coords, y_coords]).T

        # calculate and encode the control points
        self._calculate_control_points(track, track_points)

        # calculate BEZIER_SEGMENTS, CURVATURE_PROFILE, TRACK_COORDS, LENGTH
        self._calc_track_params(track)

    def fitness(self, track: abstract_track.Track) -> float:
        # calculate the percent of the track that belongs in each bin
        fitness = 100
        penalty = 0

        c = []  # track diversity, as a percent of the track within each bin

        lower_bin = self.curvature_bins[0][0]
        upper_bin = self.curvature_bins[-1][1]       
        
        for databin in self.curvature_bins:
            bin_segments = 0
            curv_min = databin[0]
            curv_max = databin[1]
            
            total_segments = 0
            # analyse
            for segment in track.CURVATURE_PROFILE:
                # ignore segements with curvatures between -0.01 and 0.01
                if -0.01 <= segment <= 0.01:
                    continue
                
                # count the segments within the current bin
                if curv_min <= segment < curv_max: bin_segments += 1 
                
                # if the segment is outside the bin range, penalise 
                if lower_bin > segment or segment > upper_bin: penalty += 20

            # ignore bins that do not contain any track segments
            if bin_segments != 0: c.append(bin_segments / total_segments)

        # calculate the entropy of curvature and normalise for 100
        c = np.asanyarray(c)  # convert to numpy array
        entropy = -np.sum(c * np.log2(c))

        fitness = (fitness * entropy) - (
            12.5 * utils.LinearAlgebra.intersection_bezier_curve(track.TRACK_COORDS)
        )
        track.encode_fitness(fitness)

        return fitness
