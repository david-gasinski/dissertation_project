#
# Generates a random track 
# and plots its curvature profile
#

import matplotlib.pyplot as plt
import numpy as np

def plot_curvature(track) -> None:
    """
        Plots the curvature profile of the track as a function of t 
            where t = [0,1], defining the interval of the entire track
            
        TODO:
            Should plot curvature vs distance, not t
    
    """
    # generate range of t values using len of self.CURVATURE_PROFILE
    interval = np.linspace(0, 1, len(track.CURVATURE_PROFILE))
    
    plt.plot(
        interval, track.CURVATURE_PROFILE
    )
    plt.title("Track")
    plt.ylabel("X")
    plt.xlabel("Y")
    plt.show()
 
