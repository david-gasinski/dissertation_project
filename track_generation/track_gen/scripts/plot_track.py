import matplotlib.pyplot as plt
import numpy as np

def plot_track(track) -> None:
    """
        Plots the curvature profile of the track as a function of t 
            where t = [0,1], defining the interval of the entire track
            
        TODO:
            Should plot curvature vs distance, not t
    
    """
    # generate range of t values using len of self.CURVATURE_PROFILE

    # convert bezier coords to numpy arrays
    bezier_coords = np.asarray(track.TRACK_COORDINATES)
    
    plt.plot(
        bezier_coords[:, 0], bezier_coords[:, 1]
    )
    plt.title("Curvature profile")
    plt.ylabel("Curvature")
    plt.xlabel("Track interval")
    plt.show()
 
