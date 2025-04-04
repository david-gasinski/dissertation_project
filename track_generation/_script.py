from track_gen.generators import convex_hull_generator
from track_gen.scripts.plot_layout_curvature import plot_track_layout
from track_gen import bezier

if __name__ == '__main__':
    gen = convex_hull_generator.ConvexHullGenerator()
    
    base_config = {
        "concave_hull": {
            "control_points" : 15,
        "bezier_interval": 0.001,
        "x_bounds": {
            "low" : -400,
            "high" : 400    
        },
        "y_bounds": {
            "low": -400,
            "high":  400
        },
        "width": 800,
        "height": 800,
        "weight_point_offset" : 50,
        "threshold_distance": 100
        }
    }
    
    track = gen.generate_track(seed=10, config=base_config["concave_hull"])
    
    # get the length of the track
    track_length = bezier.Bezier(10, 0.01).approx_arc_length(track.BEZIER_COORDINATES)
    
    print(f"""
        Track seed: {track.seed}
        Track length: {track_length}     
    """)

    plot_track_layout(track)