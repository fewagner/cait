import json
from functools import cache

class SerializingMixin:
    """
    Mixin to give serialization properties to an object. 
    
    The initializer has to be called from all children classes with all their (keyword) arguments. The mixin then saves those and if 'to_dict' is called on the child, a dictionary with the class name and the (keyword) arguments is returned. If more information is required to reconstruct a class, the 'to_dict' method may be overridden (see e.g. DataHandler).
    If reconstructing a class from a dictionary requires more than calling the initializer with the (keyword) arguments, the child class has to provide a 'from_dict' class method (see e.g. DataHandler).

    **Example:**
    ::
        import cait.versatile as vai

        class SomeClass(vai.serializing.SerializingMixin):
            def __init__(self, kwarg1=None, kwarg2=None):
                super().__init__(kwarg1=kwarg1, kwarg2=kwarg2)

        c = SomeClass()
        c.to_dict()
    """
    def __init__(self, *args, **kwargs):
        # Save (keyword) arguments for later reconstruction
        self._init_args = args
        self._init_kwargs = kwargs
        
    def to_dict(self):
        """
        Returns a dictionary representation of the object.
        """
        # Find all serializable classes
        my_subclasses = get_serializable_classes()

        # Replace serializable classes (subclasses of SerializingMixin)
        # by their dictionary representation
        args = list()
        kwargs = dict()

        for a in self._init_args:
            # If argument is a subclass, call its to_dict method
            if any([isinstance(a, msc) for msc in my_subclasses]):
                args.append(a.to_dict())
            # If argument is a list of subclasses, call their to_dict methods
            # (this just checks if the first argument in the list is a subclass and
            # assumes that all others are)
            elif isinstance(a, list) and a and any([isinstance(a[0], msc) for msc in my_subclasses]):
                args.append([x.to_dict() for x in a])
            # Else, just use the argument as is
            else:
                args.append(a)

        # Same for keyword arguments
        for k,v in self._init_kwargs.items():
            if any([isinstance(v, msc) for msc in my_subclasses]):
                kwargs[k] = v.to_dict()
            elif isinstance(v, list) and v and any([isinstance(v[0], msc) for msc in my_subclasses]):
                kwargs[k] =  [x.to_dict() for x in v]
            else:
                kwargs[k] = v
        
        return {"class": self.__class__.__name__, "args": args, "kwargs": kwargs}
    
def all_subclasses(cls):
    """
    Returns complete list of a class's subclasses (recursively).
    From https://stackoverflow.com/a/3862957
    """
    return list(set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in all_subclasses(c)]))

def get_serializable_classes():
    """
    Returns a complete (recursive) list of all classes that use the SerializingMixin, i.e. which are serializable.
    """
    return all_subclasses(SerializingMixin)

def is_valid_obj_dict(d: dict):
    """
    Returns a boolean on whether a dictionary represents a valid, deserializable class.
    """
    return all([x in d.keys() for x in ["class", "args", "kwargs"]])

def dict2obj(d: dict):
    """
    Converts a (valid) dictionary to the cait.versatile object that it represents.
    """
    if not all([x in d.keys() for x in ["class", "args", "kwargs"]]):
        raise KeyError("JSON Dictionary must contain keys ['class', 'args', 'kwargs'] to be deserialized.")
    
    # Construct a dictionary whose keys are class names and whose values
    # are the class objects
    conversion = {c.__name__: c for c in get_serializable_classes()}
    cls_name = d["class"]

    # Check if the class specified by the dictionary is known
    if cls_name not in conversion.keys():
        raise KeyError(f"{cls_name} is not a known, deserializable cait.versatile object.")
    
    # If the class provides a special from_dict method, it is used and the general 
    # procedure below is skipped
    if hasattr(conversion[cls_name], "from_dict"):
        return conversion[cls_name].from_dict(d)
    
    args, kwargs = [], {}

    # Recursively replace valid object dictionaries in args by their objects
    for a in d["args"]:
        if isinstance(a, dict) and is_valid_obj_dict(a):
            args.append(dict2obj(a))
        elif isinstance(a, list) and a and isinstance(a[0], dict) and is_valid_obj_dict(a[0]):
            args.append([dict2obj(x) for x in a])
        else:
            args.append(a)
    
    # Recursively replace valid object dictionaries in kwargs by their objects
    for k,v in d["kwargs"].items():
        if isinstance(v, dict) and is_valid_obj_dict(v):
            kwargs[k] = dict2obj(v)
        elif isinstance(v, list) and v and isinstance(v[0], dict) and is_valid_obj_dict(v[0]):
            kwargs[k] = [dict2obj(x) for x in v]
        else:
            kwargs[k] = v

    # Construct the class and return it
    return conversion[cls_name](*args, **kwargs)
    
def json_dumps(obj: SerializingMixin):
    """
    Returns the string representation of a serializable cait.versatile object (e.g. iterator or data source).

    :param obj: The object to be serialized.
    :type obj: SerializingMixin

    **Example:**
    ::
        import cait.versatile as vai

        md = vai.MockData()
        it = md.get_event_iterator()

        md_str = vai.serializing.json_dumps(md)
        it_str = vai.serializing.json_dumps(it)

        recoverd_md = vai.serializing.json_loads(md_str)
        recoverd_it = vai.serializing.json_loads(it_str)
    """
    if not isinstance(obj, SerializingMixin):
        raise TypeError("Only objects that use the SerializingMixin can be serialized. E.g. iterators and data sources.")
    return json.dumps(obj.to_dict())

@cache
def json_loads(s: str):
    """
    Returns a cait.versatile object constructed from its string representation (e.g. iterator or data source).

    :param s: The string representation of object to be deserialized.
    :type s: str

    **Example:**
    ::
        import cait.versatile as vai

        md = vai.MockData()
        it = md.get_event_iterator()

        md_str = vai.json_dumps(md)
        it_str = vai.json_dumps(it)

        recoverd_md = vai.json_loads(md_str)
        recoverd_it = vai.json_loads(it_str)
    """
    d = json.loads(s)
    if not isinstance(d, dict):
        raise TypeError("JSON string must represent a python dictionary.")
    if not is_valid_obj_dict(d):
        raise KeyError("JSON string does not represent a valid cait.versatile object. Dictionary must contain keys ['class', 'args', 'kwargs'] to be deserialized.")
    
    return dict2obj(d)