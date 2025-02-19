
def clamp(value: int, min: int, max: int):
    if value >= max:
        return min + (value - max)
    if value <= min:
        return max - (min - value)
    return value