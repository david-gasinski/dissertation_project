import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import codecs
import json

from helpers import interp_track
from helpers import calc_splines
from helpers import calc_head_curv_an

def get_curv(path: str) -> None:
    track = pd.read_csv(path).to_numpy()
              
    # need to get regular 1m samples
    # sample track at 1m intervals
    sampled_track = interp_track.interp_track(track, 1)
    
    # curvature
    # calculate curvature (required to be able to differentiate straight and corner sections)
    path_cl = np.vstack((sampled_track[:, :2], sampled_track[0, :2]))
    coeffs_x, coeffs_y = calc_splines.calc_splines(path=path_cl)[:2]
    psi, kappa_path = calc_head_curv_an.calc_head_curv_an(coeffs_x=coeffs_x,
                                                         coeffs_y=coeffs_y,
                                                         ind_spls=np.arange(0, coeffs_x.shape[0]),
                                                         t_spls=np.zeros(coeffs_x.shape[0]))
    return kappa_path

def save_np(path: str, arr: np.ndarray) -> None:
    b = arr.tolist() # nested lists with same data, indices
    json.dump(b, codecs.open(path, 'w', encoding='utf-8'), 
          separators=(',', ':'), 
          sort_keys=True, 
          indent=4)

def read_np(path: str) -> None:
    obj_text = codecs.open(path, 'r', encoding='utf-8').read()
    py_arr = json.loads(obj_text) # python arr
    return np.array(py_arr)

if __name__ == '__main__':
    path = r"C:\Users\dgasi\Desktop\workspace\environment_shaping_with_ac\dissertation_project\track_generation\datasets\racetrack-database\tracks"
    output_dir = r"C:\Users\dgasi\Desktop\workspace\environment_shaping_with_ac\dissertation_project\track_generation\datasets\scripts\output"
    
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    
    # get the curvature profile
    for track_file in files:
        curv_prof = get_curv(os.path.join(path, track_file))
        
        # save
        save_np(os.path.join(output_dir, track_file), curv_prof)
        print(f"Saved tracak {track_file}")        

