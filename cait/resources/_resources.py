import os

def get_resource_path(name: str):
    """
    Get the path of a resource stored in the library.

    TODO
    """
    return os.path.dirname(__file__) + '/' + name

def change_channel(model, new_channel: int=1):
    """TODO"""

    # feature_keys
    for i, key in enumerate(model.feature_keys):
        for c in range(10):
            if str(c) in key:
                model.feature_keys[i] = model.feature_keys[i].replace(str(c), str(new_channel))

    # label_keys
    for i, key in enumerate(model.label_keys):
        for c in range(10):
            if str(c) in key:
                model.label_keys[i] = model.label_keys[i].replace(str(c), str(new_channel))

    # norm_vals
    new_norm_vals = {}
    for k in list(model.norm_vals.keys()):
        for c in range(10):
            if str(c) in key:
                new_norm_vals[k.replace(str(c), str(new_channel))] = model.norm_vals[k]
    model.norm_vals = new_norm_vals

    # down_keys
    for i, key in enumerate(model.down_keys):
        for c in range(100):
            model.down_keys[i] = model.down_keys[i].replace(str(c), str(new_channel))

    return model