import numpy as np
import pytest

import cait.versatile as vai

from ..fixtures import datahandler, tempdir


# Tests for input validation and no errors with 
# various combinations of input arguments.
# Does NOT (yet) test for correct functioning
@pytest.mark.filterwarnings("ignore:Ill-conditioned matrix")
class TestApplyTemplateFit:
    def test_one_channel(self, datahandler):
        mock_data = vai.MockData(dt_us=datahandler.dt_us)
        mock_it = mock_data.get_event_iterator()[0]

        group = "events_tf_one_ch"

        datahandler.include_event_iterator(group, mock_it)

        datahandler.apply_template_fit(
            group=group,
            sev=mock_data.sev[0],
            bl_poly_order=3,
            truncation_limit=None,
            fit_onset=True,
        )

    def test_one_channel_of_two(self, datahandler):
        mock_data = vai.MockData(dt_us=datahandler.dt_us)
        mock_it = mock_data.get_event_iterator()

        group = "events_tf_two_ch"

        datahandler.include_event_iterator(group, mock_it)

        datahandler.apply_template_fit(
            group=group,
            sev=mock_data.sev[0],
            bl_poly_order=3,
            truncation_limit=None,
            fit_onset=True,
            only_channels=0
        )

    @pytest.mark.parametrize("correlated", [True, False])
    def test_two_channels(self, datahandler, correlated):
        mock_data = vai.MockData(dt_us=datahandler.dt_us)
        mock_it = mock_data.get_event_iterator()

        group = f"events_tf_two_ch_correlated_{correlated}"

        datahandler.include_event_iterator(group, mock_it)

        datahandler.apply_template_fit(
            group=group,
            sev=mock_data.sev,
            bl_poly_order=3,
            truncation_limit=None,
            correlated=correlated,
            fit_onset=True,
        )

    @pytest.mark.parametrize("correlated", [True, False])
    def test_two_channels_different_settings(self, datahandler, correlated):
        mock_data = vai.MockData(dt_us=datahandler.dt_us)
        mock_it = mock_data.get_event_iterator()

        group = f"events_tf_two_ch_diff_sett_correlated_{correlated}"

        datahandler.include_event_iterator(group, mock_it)

        datahandler.apply_template_fit(
            group=group,
            sev=mock_data.sev,
            bl_poly_order=[3, 1],
            truncation_limit=[0.5, 0.3],
            correlated=correlated,
            fit_onset=True,
        )

    @pytest.mark.parametrize("correlated", [True, False])
    def test_two_channels_with_flag(self, datahandler, correlated):
        mock_data = vai.MockData(dt_us=datahandler.dt_us)
        mock_it = mock_data.get_event_iterator()

        group = f"events_tf_flag_correlated_{correlated}"

        flag = np.ones(len(mock_it), dtype=bool)
        flag[::2] = False

        datahandler.include_event_iterator(group, mock_it)

        datahandler.apply_template_fit(
            group=group,
            sev=mock_data.sev,
            bl_poly_order=[3, 1],
            truncation_limit=[0.5, 0.3],
            correlated=correlated,
            fit_onset=True,
            event_flag=flag
        )

    @pytest.mark.parametrize("poly_order", [None, 1, 3, [None, 1], [0, None], [3, 1]])
    def test_two_channels_different_poly_orders(self, datahandler, poly_order):
        mock_data = vai.MockData(dt_us=datahandler.dt_us)
        mock_it = mock_data.get_event_iterator()

        datahandler.include_event_iterator(f"events_tf_poly_order_{poly_order}", mock_it)
        datahandler.include_event_iterator(f"events_tf_corr_poly_order_{poly_order}", mock_it)

        datahandler.apply_template_fit(
            group=f"events_tf_corr_poly_order_{poly_order}",
            sev=mock_data.sev,
            bl_poly_order=poly_order,
            truncation_limit=None,
            correlated=True,
            fit_onset=True,
        )

        datahandler.apply_template_fit(
            group=f"events_tf_poly_order_{poly_order}",
            sev=mock_data.sev,
            bl_poly_order=poly_order,
            truncation_limit=None,
            correlated=False,
            fit_onset=True,
        )

    def test_input_validation(self, datahandler):
        mock_data = vai.MockData(dt_us=datahandler.dt_us)
        mock_it = mock_data.get_event_iterator()

        group = "events_tf_input_validation"

        datahandler.include_event_iterator(group, mock_it)

        # Group unavailable
        with pytest.raises(KeyError):
            datahandler.apply_template_fit(group="events", sev=mock_data.sev)

        # Attempt correlated with one channel
        with pytest.raises(ValueError):
            datahandler.apply_template_fit(
                group=group, 
                sev=mock_data.sev[0],
                only_channels=0,
                correlated=True
                )
            
        # SHAPE MISMATCHES
        # channels sev < channels
        with pytest.raises(ValueError):
            datahandler.apply_template_fit(group=group, sev=mock_data.sev[0])

        # channels bl_poly_order < channels
        with pytest.raises(ValueError):
            datahandler.apply_template_fit(group=group, sev=mock_data.sev, bl_poly_order=[3])

        # channels bl_poly_order > channels
        with pytest.raises(ValueError):
            datahandler.apply_template_fit(group=group, sev=mock_data.sev, bl_poly_order=[3, 2, 1])

        # channels truncation_limit < channels
        with pytest.raises(ValueError):
            datahandler.apply_template_fit(group=group, sev=mock_data.sev, truncation_limit=[0.5])

        # channels truncation_limit > channels
        with pytest.raises(ValueError):
            datahandler.apply_template_fit(group=group, sev=mock_data.sev, truncation_limit=[0.5, 0.6, 0.7])

        # channels fit_onset < channels
        with pytest.raises(ValueError):
            datahandler.apply_template_fit(group=group, sev=mock_data.sev, fit_onset=[False])

        # channels fit_onset > channels
        with pytest.raises(ValueError):
            datahandler.apply_template_fit(group=group, sev=mock_data.sev, fit_onset=[False, True, False])

        # Overwrite warning
        datahandler.apply_template_fit(group=group, sev=mock_data.sev)
        with pytest.raises(KeyError):
            datahandler.apply_template_fit(group=group, sev=mock_data.sev)

                