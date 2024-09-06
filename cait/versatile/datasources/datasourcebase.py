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

    @property
    @abstractmethod
    def dt_us(self):
        """
        The length of a sample in the data in microseconds.
        
        :return: Microsecond time-delta
        :rtype: int
        """
        ...

    @property
    def sample_frequency(self):
        """
        The sampling frequency of the data in Hz.
        
        :return: Sampling frequency (Hz)
        :rtype: int
        """
        return int(1e6//self.dt_us)