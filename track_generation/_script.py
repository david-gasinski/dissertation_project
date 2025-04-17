from track_gen.generators import convex_hull_generator
from track_gen.scripts.plot_layout_curvature import plot_track_layout, n_plot_track_layout
from track_gen.generators.track_generator import TrackGenerator

import timeit

if __name__ == "__main__":
    gen = convex_hull_generator.ConvexHullGenerator()

    base_config = {
        "concave_hull": {
            "control_points": 10,
            "bezier_interval": 0.1,
            "x_bounds": {"low": -200, "high": 200},
            "y_bounds": {"low": -200, "high": 200},
            "width": 400,
            "height": 400,
            "control_point_offset": 20,
            "threshold_distance": 50,
            "slope_offset": {"high": 0.1, "low": -0.1},
            "render": True,
            "curvature_bin_dataset": "datasets\\scripts\\output\\data_bins\\curvature_bins.json"
        }
    }
    
    base_config_old = {
        "concave_hull": {
            "control_points": 10,
            "bezier_interval": 0.1,
            "x_bounds": {"low": -200, "high": 200},
            "y_bounds": {"low": -200, "high": 200},
            "width": 400,
            "height": 400,
            "weight_point_offset": 20,
            "threshold_distance": 50,
            "slope_offset": {"high": 0.1, "low": -0.1},
            "render": True
        }
    }
    # do some funky stuff with timeit
    start = timeit.default_timer()
    #track = gen.generate_track(seed=10, config=base_config_old["concave_hull"])
    end = timeit.default_timer() - start
    
    print("Generation took {}".format(end))
    # get the length of the track
    #track_length = bezier.Bezier(10, 0.01).arc_length(track.BEZIER_COORDINATES)

    #print(
    #    f"""
    #    Track seed: {track.seed}
    #    Track length: {track_length}     
    #"""
    #)
#
    #plot_track_layout(track)

    # do some funky stuff with timeit
    start = timeit.default_timer()

    generator = TrackGenerator(base_config['concave_hull'])
    track = generator.generate_track(432532580)
    
    end = timeit.default_timer() - start
    
    n_plot_track_layout(track)
    
    print("Generation took {}".format(end))
