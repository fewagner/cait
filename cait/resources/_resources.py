import os

def list_resources():
    """
    Print a list of all resources, that are stored in the installation of the library.
    """
    res = os.listdir(os.path.dirname(__file__))
    print('Resources stored in {}:'.format(os.path.dirname(__file__)))
    for r in res:
        if r[0] != '_' and r[0] != '.':
            print(r)

def get_resource_path(name: str):
    """
    Get the path of a resource stored in the library.

    :param name: The name of the resource. All names can be listed with list_resources.
    :type name: str
    """
    return os.path.dirname(__file__) + '/' + name

def change_channel(model, new_channel: int=1):
    """
    For a saved Lightning Module neural network, change the channel on which it acts.

    :param model: The module in which we want to change the channel.
    :type model: Lightning module
    :param new_channel: The channel to which we want to change
    :type new_channel: int
    :return: The module with changed channel.
    :rtype: Lightning module
    """

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