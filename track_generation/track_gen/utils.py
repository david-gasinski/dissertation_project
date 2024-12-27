
def clamp(value: int, min: int, max: int):
    if value > max:
        return min
    if value < min:
        return max
    return value