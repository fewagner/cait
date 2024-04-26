from abc import ABC, abstractmethod

from ..serializing import SerializingMixin

class DataSourceBaseClass(SerializingMixin, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def get_event_iterator(self, *args, **kwargs):
        ...

    @property
    @abstractmethod
    def start_us(self):
        """
        The microsecond timestamp of the start of the recording for this datasource object.
        """
        ...