from abc import ABC, abstractmethod

class DataSourceBaseClass(ABC):
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