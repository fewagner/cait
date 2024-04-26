import numpy as np

from ..datasourcebase import DataSourceBaseClass
from ...iterators.impl_mock import MockIterator
from ....fit._templates import pulse_template
from ...analysisobjects.sev import SEV
from ...analysisobjects.nps import NPS
from ...analysisobjects.of import OF

class MockData(DataSourceBaseClass):
    """
    Class to generate quick mock pulse traces (2 channels). 

    :param n_events: Number of events to simulate. Defaults to 100.
    :type n_events: int, optional
    :param record_length: Record length of the pulse traces to simulate. Defaults to 16384.
    :type record_length: int, optional
    :param dt_us: Microsecond time base of the pulse traces to simulate. Defaults to 10.
    :type dt_us: int, optional

    :return: Object providing mock data.
    :rtype: MockData
    """
    def __init__(self, 
                 n_events: int = 100,
                 record_length: int = 16384,
                 dt_us: int = 10):
        
        super().__init__(n_events=n_events, record_length=record_length, dt_us=dt_us)
        
        self._n_events = n_events
        self._record_length = record_length
        self._dt_us = dt_us

        # Random pulse heights with mean 5V
        self._phs = np.random.normal(5, 0.5, (2, n_events))
        # Random offsets between -3 and 3 V
        self._offsets = 6*np.random.rand(2, n_events) - 3
        # Random seeds to get reproducible noise traces
        self._rand_seeds = np.random.randint(0, 1000, size=n_events)
        # Fake trigger timestamps
        self._start = 1426321613000000 # March 14, 2015
        self._m_time = 10 # 10 hours measuring time
        self._ts = np.sort(np.random.randint(self._start, self._start+self._m_time*3600*1000*1000, size=n_events))
        # Record window used to evaluate the pulse model
        self._t = (np.arange(record_length) - record_length/4)*dt_us/1000

        # Pulse templates for two channels
        self._template = np.array([
            pulse_template(self._t, 0, 0.5, 0.5, 0.3*dt_us, 0.1*dt_us, 1*dt_us),
            pulse_template(self._t, 0, 0.5, 0.5, 0.3*dt_us, 0.01*dt_us, 0.4*dt_us)
        ])
        self._template = self._template/np.max(self._template, axis=-1, keepdims=True)

    def __repr__(self):
        return f'{self.__class__.__name__}(n_events={self.n_events}, record_length={self.record_length}, dt_us={self.dt_us}, measuring_time_h={self._m_time:.2f})'

    def get_event_iterator(self, batch_size: int = None):
        """
        Return an event iterator over the events in this mock data instance.

        :param batch_size: The number of events to be returned at once (these are all read together). There will be a trade-off: large batch_sizes cause faster read speed but increase the memory usage.
        :type batch_size: int

        :return: Event iterator
        :rtype: MockIterator
        """
        return MockIterator(self, batch_size=batch_size)
    
    def get_event(self, inds: int, channel: slice = None):
        """
        Return a single event for a given index.

        :param inds: The index of the event that we want to read from the mock data.
        :type inds: int
        :param channel: The channel of the event that we want to read from the mock data. If None, then all channels are returned.
        :type channel: int

        :return: Event
        :rtype: np.ndarray
        """
        inds = [inds] if isinstance(inds, int) else inds
        ph = self._phs[..., inds].T[...,None]
        off = self._offsets[..., inds].T[...,None]
        rng = [np.random.default_rng(self._rand_seeds[i]) for i in inds]
        noise = [0.1*r.standard_normal(size=self.n_channels*self._record_length)
                 for r in rng]

        out =  off + ph*self._template[None,...] + np.reshape(noise, (len(inds), self.n_channels, self._record_length))

        return out[:, channel]
    
    @property
    def sev(self):
        return SEV(self._template)

    @property
    def nps(self):
        rand = np.random.normal(size=(100, 2, self._record_length))
        nps = np.mean(np.abs(np.fft.rfft(rand))**2, axis=0)
        return NPS(nps)
    
    @property
    def of(self):
        return OF(self.sev, self.nps)

    @property
    def n_events(self):
        return self._n_events
    
    @property
    def n_channels(self):
        return 2
    
    @property
    def record_length(self):
        return self._record_length
    
    @property
    def record_window(self):
        return self._t
    
    @property
    def dt_us(self):
        return self._dt_us
    
    @property
    def start_us(self):
        return self._start
    
    @property
    def timestamps(self):
        return self._ts