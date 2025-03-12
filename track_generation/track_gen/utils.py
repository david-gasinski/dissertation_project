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
    def line_eq(slope: float, b_x: float, i_min: float, i_max: float, i: float):
        t = np.linspace(i_min, i_max, i) + b_x
        return np.column_stack(
            (t, (slope * t))
    )
    
    @staticmethod
    def line_eq_tan(slope: float, b_x: float, b_y: float, i_min: float, i_max: float, i: float) -> np.ndarray:
        t = np.linspace(i_min, i_max, i)[:, np.newaxis]
        return np.column_stack(
            (t, (slope * t) + b_y)
        )
        
    @staticmethod
    def line_eq_tan_np(slope: np.ndarray, b_x: np.ndarray, b_y: np.ndarray, i_min: float, i_max: float, i: float) -> np.ndarray:
        # broadcast to (i, 1)
        t = np.linspace(i_min, i_max, i)[:, np.newaxis]

        x = slope[:, np.newaxis]
        slope = slope[:, np.newaxis] * t.T
        b_y = b_y[:, np.newaxis]
        return np.column_stack((t + b_x, slope + b_y))
        
    @staticmethod
    def get_y_intercept(slope: Union[float, np.ndarray], y: Union[float, np.ndarray], x: Union[float, np.ndarray]) -> float:
        return y - (slope * x)
    
        