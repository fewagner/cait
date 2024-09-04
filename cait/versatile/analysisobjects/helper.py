import numpy as np

def is_array_like(data):
    """Returns True if 'data' can be successfully converted to np.ndarray."""
    try:
        arr = np.array(data)
        return True
    except ValueError:
        return False