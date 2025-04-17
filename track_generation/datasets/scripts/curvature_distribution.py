import codecs
import json
import os

import numpy as np
import matplotlib.pyplot as plt

def read_np(path: str) -> None:
    obj_text = codecs.open(path, 'r', encoding='utf-8').read()
    py_arr = json.loads(obj_text) # python arr
    return np.array(py_arr)

def save_np(path: str, arr: np.ndarray) -> None:
    b = arr.tolist() # nested lists with same data, indices
    json.dump(b, codecs.open(path, 'w', encoding='utf-8'), 
          separators=(',', ':'), 
          sort_keys=True, 
          indent=4)

if __name__ == '__main__':
    
    curv_profiles = r"C:\Users\dgasi\Desktop\workspace\environment_shaping_with_ac\dissertation_project\track_generation\datasets\scripts\output\curvature_profiles"
    files = [f for f in os.listdir(curv_profiles) if os.path.isfile(os.path.join(curv_profiles, f))]
    
    output_dir = r"C:\Users\dgasi\Desktop\workspace\environment_shaping_with_ac\dissertation_project\track_generation\datasets\scripts\output\data_bins"
    
    tracks = []    
    for track_file in files:

        # load track
        track = read_np(os.path.join(curv_profiles, track_file))
        
        tracks.append(track)

    tracks_np = np.concatenate(tracks)
    
    hist, bin_edges = np.histogram(tracks_np, bins=16)
    
    bins = np.ndarray(shape=(16,2))
    
    for idx in range(len(bin_edges) - 1):
        n_idx = idx + 1
        
        bins[idx][0] = bin_edges[idx]
        bins[idx][1] = bin_edges[n_idx]
    
    hist[8] = int(hist[8] * 0.01)    
    
    plt.bar(bin_edges[:-1], hist)
    
    plt.ylabel("Frequency") 
    plt.xlabel("Curvature (rad/m)")
    
    plt.hist()
    
    plt.title("Curvature distribution of 25 FIA Grade 1 Tracks")
    
    plt.show()
    save_np(os.path.join(output_dir, "curvature_bins.json"), bins)
    
    
    

    
