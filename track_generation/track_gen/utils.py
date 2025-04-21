import numpy as np
from typing import Union
import matplotlib.pyplot as plt
import codecs
import json 

def clamp(value: int, min: int, max: int):
    """
        min: int (inclusive)
        max: int (exclusive)
    """
    if value >= max:
        return min + (value - max)
    if value < min:
        return max - (min - value)
    return value

class LinearAlgebra:
    
    @staticmethod
    def linear_eq(slope: float, b_x: float, y_intercept: float, i_min: float, i_max: float, i: float) -> float:
        t = np.linspace(i_min, i_max, i)
        
        # if the slope is less than 1 it will cause the control points 
        # to be (x / slope) times distance apart. if lim slope -> 0 then,
        # lim (x / slope) -> inf
        # if slope > 1, normalise t to slope   
        if (np.abs(slope) > 1):
            t =  (t / slope) + b_x
        else:
            t += b_x
        y = ((t * slope) + y_intercept)
        return np.stack((t,y), axis=1)

    @staticmethod
    def deprecated_line_eq(slope: float, b_x: float, b_y: float, i_min: float, i_max: float, i: float) -> np.ndarray:
        """
            Over a given internal i across i_min, i_max
            return array of points on line with gradient slope and base b_x, b_y
        """
        t = np.linspace(i_min, i_max, i)[:, np.newaxis]
        return np.vstack(
            (t.T + b_x, (slope * (t.T + b_x)))
        )
        
    @staticmethod
    def intersection_bezier_curve(points: Union[list[float], np.ndarray]) -> bool:
        """
            Calculates any intersections along points. Returns the amount of intersections
        """
        
        # for every 2 points in the curve
        # define a line
        # test its intersection against other line combination
        # O(n^2)
        # defo a better way of doing this        
        num_points = len(points)
        intersections = 0
        
        for point in range(0, num_points):
            next_index = clamp(point + 1, 0, num_points)
            
            if (point == num_points - 1) or (next_index == 0):
                    continue
                
            _current = points[point]
            _next = points[next_index]
            
            # for all other points, find intersection 
            for intersect_test in range(next_index, num_points):
                next_intersect_index = clamp(intersect_test + 1, 0, num_points - 1)
                
                if (intersect_test == num_points - 1) or (next_intersect_index == 0):
                    break
                
                _current_intersect = points[intersect_test]
                _next_intersect = points[next_intersect_index]
                
                does_intersect = LinearAlgebra.do_segments_intersect(
                    _current, _next, _current_intersect, _next_intersect
                ) 
                
                if does_intersect:
                    intersections += 1    
                    
        return intersections
  
    @staticmethod
    def line_eq(slope: float, x: float, c: float = 0) -> float:
        return (slope * x) + c
    
        
    @staticmethod
    def get_y_intercept(slope: Union[float, np.ndarray], y: Union[float, np.ndarray], x: Union[float, np.ndarray]) -> float:
        return y - (slope * x)
    
    @staticmethod 
    def calculate_slopes(points: np.ndarray, origin_x: float = 0, origin_y: float = 0) -> Union[float, np.ndarray]:        
        # calculate slope, y / x
        return (points[:, 1] - origin_y) / (points[:, 0] - origin_x) 

    @staticmethod
    def calculate_slope_tangent(slopes: np.ndarray) -> np.ndarray:
        return - ( 1 / slopes )       
    
    @staticmethod
    def euclidean_distance(p1: Union[list[float], np.ndarray], p2: Union[list[float], np.ndarray]) -> float:
        # convert to numpy array
        p1 = np.asarray(p1)
        p2 = np.asarray(p2)            
        
        return np.linalg.norm(p1 - p2)

    @staticmethod
    def manhanttan_distance(p1: Union[list[float], np.ndarray], p2: Union[list[float], np.ndarray]) -> float:
        p1 = np.asarray(p1)
        p2 = np.asarray(p2)
            
        return np.absolute((p1[0] - p2[0]) + (p1[1] - p2[1]))
    
    @staticmethod
    def offset_xy(offset_dist: float, flip_sign: bool = False) -> list[float]:
        """
            Converts a fixed distance offset into equal x and y offsets using the formulat
            a^2 + b^2 = c^2 rearranged as
            
            b = sqrt(c^2 / 2)
            
            as this is used for a distance measure, flip_sign can be used to return a negative offset
        """
        _offset = np.sqrt(offset_dist / 2)
        return _offset if not flip_sign else -_offset
        
    @staticmethod
    def get_intersect(a1: np.ndarray, a2: np.ndarray, b1: np.ndarray, b2: np.ndarray) -> Union[np.ndarray, None]:
        """
        #### Returns point of intersection of line a1->a2 and b1->b2
        
        code from https://stackoverflow.com/questions/3252194/numpy-and-line-intersections
        """
        # stack the coordinates
        s = np.vstack([a1,a2,b1,b2])
        
        # convert to homogenous coordinates
        h = np.hstack((s, np.ones((4,1))))
        
        # find the first line vector by getting the cross product of the homogenous coordinates
        l1 = np.cross(h[0], h[1])
        
        # find the second line vector by getting the cross product of the homogenous coordinates
        l2 = np.cross(h[2], h[3])
        
        # find the point of intersection using the cross product
        x, y, z = np.cross(l1 , l2)
        
        # if z == 0, then lines are parallel
        if (z == 0):
            return np.asarray([float('inf'), float('inf')])
        
        return np.asarray([x/z, y/z])
    
    @staticmethod
    def do_segments_intersect(a1: Union[list[float, np.ndarray]], a2: Union[list[float, np.ndarray]], b1: Union[list[float, np.ndarray]], b2: Union[list[float, np.ndarray]]) -> bool:
        """
        Test if two line segments defined by two points each intersect.
        
        #### Returns point of intersection of line a1->a2 and b1->b2
        
        """
        # convert to numpy arrays
        a1 = np.asarray(a1)
        a2 = np.asarray(a2)

        b1 = np.asarray(b1)
        b2 = np.asarray(b2)
        
        # Direction vectors of the lines
        dir1 = a2 - a1
        dir2 = b2 - b1

        # Solve for t and s in the equation: P1 + t * dir1 = Q1 + s * dir2
        A = np.vstack([dir1, -dir2]).T
        b = b1 - a1

        try:
            t, s = np.linalg.solve(A, b)
            # Check if t and s are within the segment bounds [0, 1]
            if 0 <= t <= 1 and 0 <= s <= 1:
                # check none of the points are shared
                if not ((a1 == b1).all() or (a1 == b2).all() or (a2 == b1).all() or (a2 == b2).all()): 
                    return True  # Segments intersect
            else:
                return False  # Intersection is outside the segments
        except np.linalg.LinAlgError:
            # No solution exists (lines do not intersect)
            return False
  
  
class PolynomialAlgebra:
    
    @staticmethod
    def quadratic_formula(a: float, b: float, c: float) -> list[float]:
        # dont return complex numbers
        poly = np.polynomial.Polynomial([a, b, c])        

        roots = poly.roots()
        if (roots.dtype == np.complex128):
            # roots are zero
            roots = np.asanyarray([0.0,0.0])
        return roots
  
    
class PerlinNoise:
    
    # noise functions
    # generates a perlin noise value for a specific grid position
    # noise values are normalised to be between values specified 
    @staticmethod
    def _perlin_noise(x, y, noise_seed=0):
        np.random.seed(noise_seed) 
        # create a array of random values from 1-256, these are the permutations
        perlin_noise = np.arange(256, dtype=int)
        np.random.shuffle(perlin_noise)

        # create a 2d array and flatten to 1D
        perlin_noise = np.stack([perlin_noise,perlin_noise]).flatten()

        # coordinates of top left
        x0, y0 = x.astype(int), y.astype(int)

        # normalised coordinates (to internal grid)
        xN, yN = x - x0, y - y0

        # calculate the fade factors based on the fade equation 
        # 6t^5 - 15t^4 + 10t^3
        u, v = PerlinNoise.fade(xN), PerlinNoise.fade(yN)

        # calculate the perlin noise components
        # each component is the result of a constant vector h, and the dot product of the vector from 
        # each corner of the grid to (x,y)
        # https://en.wikipedia.org/wiki/Perlin_noise#/media/File:PerlinNoiseGradientGrid.svg

        noise00 = PerlinNoise.gradient(perlin_noise[perlin_noise[x0] + y0], xN, yN)
        noise01 = PerlinNoise.gradient(perlin_noise[perlin_noise[x0] + y0 + 1], xN, yN -1)
        noise10 = PerlinNoise.gradient(perlin_noise[perlin_noise[x0 + 1] + y0], xN - 1, yN)
        noise11 = PerlinNoise.gradient(perlin_noise[perlin_noise[x0 + 1] + y0 + 1], xN-1, yN-1)

        # interpolate values between the noise using the fade values
        x1 = PerlinNoise.linear_interp(noise00, noise01, u)    
        x2 = PerlinNoise.linear_interp(noise10, noise11, u)
        return PerlinNoise.linear_interp(x1, x2, v) 

    @staticmethod
    def perlin_noise_scaled(size, scale,  amplitude, seed=0):
        # generate empty array
        perlin = np.full((size, size), 1)

        linear_spacing = np.linspace(0, amplitude, size)
        x,y = np.meshgrid(linear_spacing, linear_spacing)

        return (PerlinNoise._perlin_noise(x,y, seed) / amplitude) * scale + perlin 
    
    @staticmethod
    def fade(t):
        return 6*t**5 - 15*t**4 + 10*t**3

    @staticmethod
    def gradient(h, x, y):
        # converts the value of h into a gradient vector and returns the dot product with vector (x,y)
        # https://en.wikipedia.org/wiki/Perlin_noise#/media/File:PerlinNoiseGradientGrid.svg
        vectors = np.array([[0,1], [0, -1], [1,0], [-1,0]]) # possible values of noise
        h = vectors[h % 4]
        return h[:, :, 0] * x + h[:, :, 1] * y

    @staticmethod
    def linear_interp(a, b, x):
        return a + x * (b-a)
    
    
def convert_to_screen(coords: list[float]) -> list[float]:
    return [coords[0] + 400, coords[1] + 400]

def convert_to_screen_scalar(coord: float) -> float:
    return coord + 400

def read_np(path: str) -> None:
    """
        Open a file containg a serialized numpy array within
        Load the numpy array and return 
    """
    obj_text = codecs.open(path, 'r', encoding='utf-8').read()
    py_arr = json.loads(obj_text) # python arr
    return np.array(py_arr)

def save_track(track, data_path, img_path):
    """
        Exports a given track to a json file and a .png image
    """
    json.dump(track.serialize(), codecs.open(data_path, 'w', encoding='utf-8'), 
          separators=(',', ':'), 
          sort_keys=True, 
          indent=4)
    
    # plot the track 
    plt.plot(track.TRACK_COORDS[:, 0], track.TRACK_COORDS[:, 1])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Fitness: {track.fitness()} [{track.seed}]")
    plt.savefig(img_path)
    plt.clf()