from typing import Union
from deprecation import deprecated

import numpy as np

import cait as ai
import cait.versatile as vai

from ..styles._print_styles import txt_fmt


class MainParametersMixin:
    def cmp(self,
            group: str = 'events',
            **kwargs,
            ):
        """
        Calculate main parameters of the group specified by `group`.

        See :class:`cait.versatile.MainParameters` for a description of the parameters
        calculated here.

        :param group: The group for which the main parameters are calculated,
            e.g. "events", "testpulses", "noise", etc.  Defaults to "events".
        :type group: str, optional

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
        events = self.get_event_iterator(group, batch_size=100)

        print(txt_fmt('Calculating main parameters ...', style="bold"))
        mp = vai.MainParameters(self.dt_us)
        out = vai.apply(mp, events)

        # Swap array axes to be able to unpack the parameters and insert them
        # into the data handler by name.
        # vai.apply returns an array of shape (n_parameters, n_events, n_channels)
        # the following line swaps this to (n_parameters, n_channels, n_events).
        out = np.swapaxes(out, 1, 2)

        for n, t, d in zip(mp.names, mp.types, out):
            self.set(
                    group,
                    **{n: d},
                    dtype=t,
                    overwrite_existing=True,
                    write_to_virtual=False,
                    )


