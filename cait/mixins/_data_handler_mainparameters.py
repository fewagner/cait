from typing import Union, List, Callable
from deprecation import deprecated

import numpy as np

import cait as ai
import cait.versatile as vai

from ..styles._print_styles import txt_fmt


class MainParametersMixin:
    def cmp(self,
            group: str = 'events',
            with_processing: List[Callable] = [],
            tag: str = None,
            batch_size: int = 2,
            **kwargs,
            ):
        """
        Calculate main parameters of the group specified by `group`.

        See :class:`cait.versatile.MainParameters` for a description of the parameters
        calculated here.

        :param group: The group for which the main parameters are calculated,
            e.g. "events", "testpulses", "noise", etc.  Defaults to "events".
        :type group: str, optional

        :param with_processing: Optional processing to apply to each event before
            :class:`cait.versatile.MainParameters` is applied.  See
            :func:`cait.versatile.iterators.IteratorBaseClass.add_processing`.
        :type with_processing: callable or list of callable, optional

        :param tag: Optional suffix to append to the names normally written to
            the DataHandler.  Useful e.g. in conjunction with `with_processing` or `kwargs`
            arguments to separate different passses.  The suffix is separated from the
            name automatically by an underscore, i.e. `tag=fqlc` will result in datasets
            such as `pulse_height_fqlc`, `onset_fqlc`, etc.
        :type tag: str, optional

        :param batch_size: Override the default batch size of 2 when using
            :func:`cait.versatile.apply`.  May improve speed in certain circumstances.
        :type batch_size: int, optional

        :param kwargs: Keyword arguments to pass to :class:`cait.versatile.MainParameters`.
        :type kwargs: Any

        .. code-block:: python

            import cait as ai
            import cait.versatile as vai

            # Create an empty datahandler
            dh = ai.DataHandler(nmbr_channels = 2)
            dh.set_filepath(path_h5='.', fname='mock', appendix=False)
            dh.init_empty()

            # Create mock data and add it to the handler
            it = vai.MockData(record_length=2**14).get_event_iterator()
            dh.include_event_iterator("events", it, copy_events=False)

            # CMP for group "events" with DataHandler's dt_us
            dh.cmp()

            # CMP for group "testpulses"
            dh.cmp("testpulses")
        """
        events = self.get_event_iterator(group, batch_size=batch_size).with_processing(with_processing)

        mp = vai.MainParameters(self.dt_us, **kwargs)
        out = vai.apply(mp, events, pb_prefix='Calculating main parameters')

        for n, t, d in zip(mp.names, mp.types, out):
            self.set(
                    group,
                    **{f"{prefix}_{n}" if prefix is not None else n: np.atleast_2d(d.T)},
                    dtype=t,
                    overwrite_existing=True,
                    write_to_virtual=False,
                    )


