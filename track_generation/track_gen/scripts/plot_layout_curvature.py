
import matplotlib.pyplot as plt
import numpy as np

def plot_track_layout(track) -> None:
    """
        Old track generation method
    """
    
    interval = np.linspace(0, 1, len(track.CURVATURE_PROFILE))
    bezier_coords = np.asarray(track.BEZIER_COORDINATES)
    
    fig = plt.figure()
    sub_plots = fig.subplots(1,2, squeeze=False)
    
    sub_plots[0, 0].plot(bezier_coords[:, 0], bezier_coords[:, 1])
    sub_plots[0,1].plot(interval, track.CURVATURE_PROFILE) 

    fig.show() 
    plt.show()  

def n_plot_track_layout(track) -> None:
    """
        New track generation method
    """

    fig = plt.figure()
    
    sub_plots = fig.subplots(1,2, squeeze=False)
        
    interval = np.linspace(0, int(track.LENGTH) + 1, int(track.LENGTH))
                
    sub_plots[0, 1].plot(interval, track.CURVATURE_PROFILE) 
    sub_plots[0, 0].plot(track.TRACK_COORDS[:, 0], track.TRACK_COORDS[:, 1])
    
    fig.show() 
    plt.show()
    
    
    
    