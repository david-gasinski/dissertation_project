import numpy as np
from typing import Union

def clamp(value: int, min: int, max: int):
    if value >= max:
        return min + (value - max)
    if value <= min:
        return max - (min - value)
    return value

class LinearAlgebra:
    
    @staticmethod
    def line_eq(slope: float, b_x: float, b_y: float, i_min: float, i_max: float, i: float) -> np.ndarray:
        """
            Over a given internal i across i_min, i_max
            return array of points on line with gradient slope and base b_x, b_y
        """
        t = np.linspace(i_min, i_max, i)[:, np.newaxis]
        return np.vstack(
            (t.T + b_x, (slope * t.T) + b_y)
        )
        
    @staticmethod
    def line_eq_np(slope: np.ndarray, b_x: np.ndarray, b_y: np.ndarray, i_min: float, i_max: float, i: float) -> np.ndarray:
        """
            Over a given internal i across i_min, i_max
            return array of points on line with gradient slope and base b_x, b_y

            identical operation to LinearAlgebra.line_eq but supporting multiple slopes
        """
        # broadcast to (i, 1)
        t = np.linspace(i_min, i_max, i)[:, np.newaxis] 

        # broadcast slope to (10,1), transpose t to (1, 200) and add y coords
        y = slope[:, np.newaxis] * t.T + b_y[:, np.newaxis]
        x = t.T + b_x[:, np.newaxis]
        
        # add new axis to transpose on 
        x = x[:,:, np.newaxis]
        y = y[:,:, np.newaxis]
        return np.concatenate((x, y), axis=2)
        
    @staticmethod
    def get_y_intercept(slope: Union[float, np.ndarray], y: Union[float, np.ndarray], x: Union[float, np.ndarray]) -> float:
        return y - (slope * x)
    
    @staticmethod 
    def calculate_slopes(points: np.ndarray) -> np.ndarray:
        # calculate slope, y / x
        return points[:, 1] / points[:, 0] 

    @staticmethod
    def calculate_slope_tangent(slopes: np.ndarray) -> np.ndarray:
        return - ( 1 / slopes )        
    
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