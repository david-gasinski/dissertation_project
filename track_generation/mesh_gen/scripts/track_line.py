import numpy as np
import bpy
import codecs
import json
import os

CONFIG_PATH = ""

def read_np(path: str) -> None:
    """
        Open a file containg a serialized numpy array within
        Load the numpy array and return 
    """
    obj_text = codecs.open(path, 'r', encoding='utf-8').read()
    py_arr = json.loads(obj_text) # python arr
    return np.array(py_arr)

def read_config() -> dict:
    config = codecs.open(CONFIG_PATH, 'r', encoding='utf-8').read()
    return json.loads(config)

def save_config(config: dict) -> None:
    json.dump(
        config,
        codecs.open(CONFIG_PATH, 'w', encoding='utf-8'),
            separators=(',', ':'),
            sort_keys=True,
            indent=4
    ) 

class TrackLine():
    """
        This class renders a blender curve from a set of coordinates
        It also stores references to the object   
        
        https://blender.stackexchange.com/questions/180148/create-curve-from-numpy-array-using-python/180184#180184     
    """
    
    def __init__(self, mesh_name: str, coordinates: np.ndarray, weights: np.ndarray = None, closed: bool = True) -> None:
        self.coordinates = coordinates
        self.mesh_name = mesh_name
        self.weight = weights if weights else np.ones(len(self.coordinates))
        self.closed = closed
        
        self.spline = None
        self.obj = None
        
    def create_spline(self) -> None:
        # create a curve in blender
        curve_data = bpy.data.curves.new(name=self.mesh_name, type='CURVE')
        curve_data.dimensions = '3D'
        
        # add a poly spline and create len(coordinates) points
        spline = curve_data.splines.new(type='POLY')            
        spline.points.add(len(self.coordinates) - 1)
        
        x = self.coordinates[:, 0]
        y = self.coordinates[:, 1]
        z = np.zeros(len(self.coordinates))
     
        # add the points
        spline.points.foreach_set("co", self._flatten(x,y,z, self.weight))
        
        if self.closed:
            spline.use_cyclic_u = self.closed
            spline.use_cyclic_v= self.closed

        self.curve_data = curve_data
        
    def link_obj(self) -> None:
        self.obj = bpy.data.objects.new(name=self.mesh_name, object_data=self.curve_data)
        bpy.context.collection.objects.link(self.obj)
            
    def _flatten(self, *args) -> np.ndarray:
        """
            Flattens the arguments into a singuler numpy array of size sum(args.size)
        """
        c = np.empty(sum(arg.size for arg in args))
        l = len(args)
        for i, arg in enumerate(args):
            c[i::l] = arg
        return c

config = read_config()
track_obj = read_np(config['track'])            
   
track_coords = track_obj['track_coords'] # read the track from the defined config 
#track = read_np(r"C:\Users\dgasi\Desktop\workspace\environment_shaping_with_ac\dissertation_project\track_generation\test_track.json")

name = "track_path"
track = TrackLine(name, track_coords, closed=True)
track.create_spline()
track.link_obj()

# add curve modifier and append curve mesh
bpy.ops.object.modifier_add(type='CURVE')
bpy.context.object.modifiers["Curve"].object = bpy.data.objects[name] # replace with name

# to array modifier
# select the option "fit to curve"
# and append object
bpy.context.object.modifiers["Array"].fit_type = 'FIT_CURVE' # change type to fit curve
bpy.context.object.modifiers["Array"].curve = bpy.data.objects[name] # assign curve

filepath = os.path.join(os.path.join(config['blend']), f"{name}.blend") 

# save blend file
bpy.ops.wm.save_as_mainfile(filepath=filepath)

