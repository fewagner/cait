import json
from functools import lru_cache


class SerializingMixin:
    """
    Mixin to give serialization properties to an object. 
    
    The initializer has to be called from all child classes with all their (keyword) arguments. The mixin then saves those and if 'to_dict' is called on the child, a dictionary with the class name and the (keyword) arguments is returned. If more information is required to reconstruct a class, the 'to_dict' method may be overridden (see e.g. DataHandler).
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
        Return a dictionary representation of the object.
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
    
    @classmethod
    def from_dict(cls, d: dict):
        """
        Construct an object from a dictionary.
        """
        if not is_valid_obj_dict(d):
            raise KeyError("JSON dictionary must contain keys ['class', 'args', 'kwargs'] to be deserialized.")
        
        if cls.__name__ != d["class"]:
            raise Exception(f"Attempted to construct dictionary representation of '{d['class']}' from class '{cls.__name__}'.")
        
        args, kwargs = [], {}

        # Recursively replace valid object dictionaries in args by their objects
        for a in d["args"]:
            if isinstance(a, dict) and is_valid_obj_dict(a):
                args.append(load(a))
            elif isinstance(a, list) and a and isinstance(a[0], dict) and is_valid_obj_dict(a[0]):
                args.append([load(x) for x in a])
            else:
                args.append(a)
        
        # Recursively replace valid object dictionaries in kwargs by their objects
        for k,v in d["kwargs"].items():
            if isinstance(v, dict) and is_valid_obj_dict(v):
                kwargs[k] = load(v)
            elif isinstance(v, list) and v and isinstance(v[0], dict) and is_valid_obj_dict(v[0]):
                kwargs[k] = [load(x) for x in v]
            else:
                kwargs[k] = v

        # Construct the class and return it
        return cls(*args, **kwargs)
    
    def to_str(self):
        """
        Return a string representation of the object.
        """
        return json.dumps(self.to_dict())

    @classmethod
    def from_str(cls, s: dict):
        """
        Construct an object from a string.
        """
        return cls.from_dict(json.loads(s))

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

def dump(obj: SerializingMixin):
    """
    Return the dictionary representation of a serializable cait object (e.g. iterator or data source).
    """
    if not isinstance(obj, SerializingMixin):
        raise TypeError(f"Only classes inherited from SerializingMixin can be converted to dictionaries. Got {type(obj)}")
    
    return obj.to_dict()

def load(d: dict):
    """
    Converts a (valid) dictionary to the cait object that it represents.
    """
    if not is_valid_obj_dict(d):
        raise KeyError("JSON dictionary must contain keys ['class', 'args', 'kwargs'] to be deserialized.")
    
    # Construct a dictionary whose keys are class names and whose values
    # are the class objects
    conversion = {c.__name__: c for c in get_serializable_classes()}
    cls_name = d["class"]

    # Check if the class specified by the dictionary is known
    if cls_name not in conversion.keys():
        raise KeyError(f"{cls_name} is not a known, deserializable cait object.")
    
    return conversion[cls_name].from_dict(d)
    
def dumps(obj: SerializingMixin):
    """
    Return the string representation of a serializable cait object (e.g. iterator or data source).

    :param obj: The object to be serialized.
    :type obj: SerializingMixin

    **Example:**
    ::
        import cait.versatile as vai

        md = vai.MockData()
        it = md.get_event_iterator()

        md_str = dumps(md)
        it_str = dumps(it)

        recoverd_md = loads(md_str)
        recoverd_it = loads(it_str)
    """
    return json.dumps(dump(obj))

@lru_cache(maxsize=None)
def loads(s: str):
    """
    Returns an object constructed from its string representation (e.g. iterator or data source).

    :param s: The string representation of object to be deserialized.
    :type s: str

    **Example:**
    ::
        import cait.versatile as vai

        md = vai.MockData()
        it = md.get_event_iterator()

        md_str = dumps(md)
        it_str = dumps(it)

        recoverd_md = loads(md_str)
        recoverd_it = loads(it_str)
    """
    try:
        d = json.loads(s)
    except json.JSONDecodeError as e:
        # Raise an easier to understand exception
        raise TypeError("String must represent a python dictionary.") from e
    
    return load(d)