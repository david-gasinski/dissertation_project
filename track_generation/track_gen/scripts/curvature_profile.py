#
# Generates a random track 
# and plots its curvature profile
#

import matplotlib.pyplot as plt
import numpy as np

from .. import bezier
from ..generators import convex_hull_generator
from ..tracks import convex_hull_track
from .. import utils

def plot_curvature(track: convex_hull_track.ConvexHullTrack) -> None:
    """
        Plots the curvature profile of the track as a function of t 
            where t = [0,1], defining the interval of the entire track
            
        TODO:
            Should plot curvature vs distance, not t
    
    """
    # get track control points
    track_control = track.CONTROL_POINTS
    curvature = []
    
    for index in range(track._control_points):
        current_control = index
        next_control = utils.clamp(index + 1, 0, track._control_points)
                
        pass
    
    plt.show()
    return  
