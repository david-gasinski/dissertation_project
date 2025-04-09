
import matplotlib.pyplot as plt
import numpy as np

def plot_track_layout(track) -> None:
    
    interval = np.linspace(0, 1, len(track.CURVATURE_PROFILE))
    bezier_coords = np.asarray(track.BEZIER_COORDINATES)
    
    fig = plt.figure()
    sub_plots = fig.subplots(1,2, squeeze=False)
    
    sub_plots[0, 0].plot(bezier_coords[:, 0], bezier_coords[:, 1])
    sub_plots[0,1].plot(interval, track.CURVATURE_PROFILE) 

    fig.show() 
    plt.show()  

def n_plot_track_layout(track) -> None:
    fig = plt.figure()
    
    sub_plots = fig.subplots(1,2, squeeze=False)
    bezier_coords = np.asarray(track.BEZIER_COORDINATES)
        
    # flatten
    flat_curv = [x for xs in track.CURVATURE_PROFILE for x in xs]
    
    flat_curv = np.asanyarray(flat_curv)
    interval = np.linspace(0, 1, len(flat_curv))
                
    sub_plots[0, 1].plot(interval, flat_curv) 
    sub_plots[0, 0].plot(bezier_coords[:, 0], bezier_coords[:, 1])
    
    fig.show() 
    plt.show()
    
    
    
    