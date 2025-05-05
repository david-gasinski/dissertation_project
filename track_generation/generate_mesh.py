import subprocess
from tkinter import *
from tkinter import filedialog
import matplotlib.pyplot as plt
import codecs
import json

import numpy as np

from track_gen.generators import track_generator
from track_gen.tracks import convex_hull_track

CMD = [r'blender', r"--background", r"C:\Users\dgasi\Desktop\workspace\environment_shaping_with_ac\dissertation_project\track_generation\mesh_gen\blend\base.blend", r"--python" , r"C:\Users\dgasi\Desktop\workspace\environment_shaping_with_ac\dissertation_project\track_generation\mesh_gen\scripts\track_line.py"]
BLENDER_PROCESS = r'C:\Program Files (x86)\Steam\steamapps\common\Blender'

# configs
DESTINATION_CONFIG = "mesh_gen\\config.json"
TEMP_CONFIG = "C:\\Users\\dgasi\\Desktop\\workspace\\environment_shaping_with_ac\\dissertation_project\\track_generation\\mesh_gen\\temp\\track_mesh.json"

TRACK_FILENAME = None

# util functions
def read_json(path: str) -> None:
    obj_text = codecs.open(path, 'r', encoding='utf-8').read()
    return json.loads(obj_text) 

def save_json(config: dict, path: str) -> None:
    json.dump(config, codecs.open(path, 'w', encoding='utf-8'), 
          separators=(',', ':'), 
          sort_keys=True, 
          indent=4)

def run_blender() -> None:
    if TRACK_FILENAME is None:
        label_file_explorer.configure(text="No track selected")
    
    process = subprocess.call(CMD, shell=True, cwd=BLENDER_PROCESS)
    
    progress_text = f"Ran blender process. Return code {process}"
    label_file_explorer.configure(text=progress_text)

def configure_timings() -> None:
    if TRACK_FILENAME is None:
        label_file_explorer.configure(text="No track selected")

    genotype = None
    track_length = None
    seed = None
    try:
        track_conf = read_json(TRACK_FILENAME)
        genotype = np.asanyarray(track_conf["genotype"])
        track_length = track_conf["length"]
        seed = track_conf['seed']
    except FileNotFoundError as e:
        label_file_explorer.configure(text="File doesnt exist")
    except KeyError as e:
        label_file_explorer.configure(text="Not a valid track format. Must contain 'genotype', 'length' and 'seed' keys")
        
    # define timing objects
    timing_objects = [
        "AC_START_0",
        "AC_START_1",
        "AC_PIT_0",
        "AC_PIT_1",
        "AC_HOTLAP_START_0",
        "AC_TIME_0_L",
        "AC_TIME_0_R",
        "AC_TIME_1_L",
        "AC_TIME_1_R",
    ]
    # load config
    conf = read_json('config.json')
    
    # create a generator object
    track_gen = track_generator.TrackGenerator(conf['concave_hull'])
    
    # create track object
    track = convex_hull_track.ConvexHullTrack(len(genotype), seed)

    # encode control points
    track.encode_control_points(
        genotype[:, 0, np.newaxis],
        genotype[:, 1, np.newaxis],
        genotype[:, 2, np.newaxis],
        genotype[:, 3, np.newaxis],
        genotype[:, 4, np.newaxis],
        genotype[:, 5, np.newaxis],
        genotype[:, 6, np.newaxis]
    )
    
    # generate track coords
    track_gen._calculate_bezier(track)
    track_gen._track_coordinates(track, interval=0.02)
    
    track_mesh_config = {}
    default_z = 2 # 2m above the ground
    
    for timing in timing_objects:
        loc = get_coords_from_track(track, f"Select track coordinate for timing point {timing}")
        
        # encode location
        track_mesh_config[timing] = [*loc, default_z]
    
    # encode track config
    track_mesh_config['track_coords'] = track.TRACK_COORDS.tolist()        

    save_json(track_mesh_config, TEMP_CONFIG)
    label_file_explorer.configure(text="Saved mesh config, ready to generate mesh")
 
def get_coords_from_track(track, title: str = "Track"):
    ev = None
    def onclick(event):
        nonlocal ev
        ev = event
        print(f"x:{ev.xdata} y:{ev.ydata}")
        plt.close(None)

    fig, ax = plt.subplots()
    
    # convert bezier coords to numpy arrays
    bezier_coords = np.asarray(track.TRACK_COORDS)
    
    ax.scatter(
        bezier_coords[:, 0], bezier_coords[:, 1], s=113.1
    )
    ax.set_title(title)
    ax.set_ylabel("y")
    ax.set_xlabel("x")    
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()    
       
    return (ev.xdata, ev.ydata) if ev is not None else None

def plot_track(track, title: str = "Track") -> None:
    """
        Plots the curvature profile of the track as a function of t 
            where t = [0,1], defining the interval of the entire track
            
        TODO:
            Should plot curvature vs distance, not t
    
    """
    # generate range of t values using len of self.CURVATURE_PROFILE

    # convert bezier coords to numpy arrays
    bezier_coords = np.asarray(track.TRACK_COORDS)
    
    plt.scatter(
        bezier_coords[:, 0], bezier_coords[:, 1], s=113.1
    )
    plt.title(title)
    plt.ylabel("y")
    plt.xlabel("x")
    plt.show()

def browse_file() -> str:
    global TRACK_FILENAME
    filename = filedialog.askopenfilename(
        initialdir="C:\\Users\\dgasi\\Desktop\\workspace\\environment_shaping_with_ac\\dissertation_project\\track_generation\\tracks",
        title="Select a track file",
        filetypes=(("Json files", "*.json"), ("All files", "*.*")),
    )
    TRACK_FILENAME = filename
    label_file_explorer.configure(text="Opened: " + filename)


# set up window
window = Tk()
window.title("Track Mesh Generator")
window.geometry("700x700")
window.config(background="white")

# Create a File Explorer label
label_file_explorer = Label(
    window,
    text="Select a track to generate a blender mesh",
    width=100,
    height=4,
    fg="blue",
)

# set up basic functionality
button_select_track = Button(window, text="Select track file", command=browse_file)
button_configure_timing = Button(
    window, text="Configure timing positions", command=configure_timings
)
button_generate_mesh = Button(window, text="Create mesh", command=run_blender)
button_exit = Button(window, text="Exit", command=exit)

# arrange on the screen
label_file_explorer.grid(column=1, row=1)
button_select_track.grid(column=1, row=2)
button_configure_timing.grid(column=1, row=3)
button_generate_mesh.grid(column=1, row=4)
button_exit.grid(column=1, row=5)

# Let the window wait for any events
window.mainloop()
