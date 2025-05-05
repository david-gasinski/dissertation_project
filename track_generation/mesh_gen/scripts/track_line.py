import numpy as np
import bpy
import codecs
import json
import os

CONFIG_PATH = "C:\\Users\\dgasi\\Desktop\\workspace\\environment_shaping_with_ac\\dissertation_project\\track_generation\\mesh_gen\\config.json"

def read_config(path: str) -> dict:
    config = codecs.open(path, 'r', encoding='utf-8').read()
    return json.loads(config)

def save_config(config: dict, path: str) -> None:
    json.dump(
        config,
        codecs.open(path, 'w', encoding='utf-8'),
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

config = read_config(CONFIG_PATH)
track_conf = read_config(config['track'])     # read the track from the defined config 

track_coords = np.asanyarray(track_conf['track_coords'])
name = "track_path"
track = TrackLine(name, track_coords, closed=True)
track.create_spline()
track.link_obj()

# add curve modifier and append curve mesh
objects = ["1GRASS_ground", "2ROAD_track", "3KERB_kerb"]

curve = bpy.data.objects[name]
for obj in objects:
    bpy.context.view_layer.objects.active = bpy.data.objects[obj]

    # create curve modifier
    bpy.ops.object.modifier_add(type='CURVE')
    bpy.context.object.modifiers["Curve"].object = bpy.data.objects[name] # replace with name
    
    # change curve modifer to FIT_CURVE flag
    bpy.context.object.modifiers["Array"].fit_type = 'FIT_CURVE' # change type to fit curve
    
    # assign curve to modifier
    bpy.context.object.modifiers["Array"].curve = curve
    
    # Assign decimate modifier to reduce face count
    # but only onto ground mesh or kerb
    if obj == "1GRASS_ground":
        bpy.ops.object.modifier_add(type='DECIMATE')
        bpy.context.object.modifiers["Decimate"].ratio = 0.075
    elif obj == "3KERB_track":
        bpy.ops.object.modifier_add(type='DECIMATE')
        bpy.context.object.modifiers["Decimate"].ratio = 0.55
        
    # convert to mesh
    bpy.ops.object.convert(target='MESH')
    
# deselect all objects
bpy.ops.object.select_all(action='DESELECT')

# select curve and delete 
bpy.context.view_layer.objects.active = bpy.data.objects[name]
bpy.ops.object.delete()

# building the timing objects
timing_objects = [
    "AC_START_0",
    "AC_START_1",
    "AC_PIT_0",
    "AC_PIT_1",
    "AC_HOTLAP_START_0",
    "AC_TIME_0_L",
    "AC_TIME_0_R",
    "AC_TIME_1_L",
    "AC_TIME_1_R"
]

def spawn_volume(name: str, pos: list[float, float, float]) -> None:
    bpy.ops.object.volume_add(align='WORLD', location=(pos[0], pos[1], pos[2]), scale=(1, 1, 1))
    bpy.context.object.name = name
    
    # ensure object Y is facing up and Z is forward
    bpy.context.scene.transform_orientation_slots[0].type = 'LOCAL' # switch to local coordinate mode
    bpy.context.object.rotation_euler[0] = 1.5708 # apply 90deg rotation

# spawn the timing objects    
for timing in timing_objects:
    
    spawn_volume(timing, track_conf[timing])
    
filepath = os.path.join(os.path.join(config['blend']), f"{name}.blend") 

# save blend file
bpy.ops.wm.save_as_mainfile(filepath=filepath)

