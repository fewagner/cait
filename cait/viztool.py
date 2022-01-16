import numpy as np

import plotly.graph_objs as go
import plotly.offline as py
import pandas as pd  # new to cait
import numpy as np
import ipywidgets as widgets  # new to cait
from ipywidgets import interactive, HBox, VBox, Button, Layout
import h5py
from IPython.display import display

from .data_handler import DataHandler


class VizTool():
    """
    A class to visualize the parameters and corresponding events, and allow for interactive cuts.

    The VizTool can get the data either from a CSV file, or from an HDF5 file. Only in the second case, the calculation
    of standard event and visualization of raw data is possible.

    :param csv_path: The full path to the CSV file.
    :type csv_path: str
    :param nmbr_features: The number of columns in the CSV file, which is also the number of features.
    :type nmbr_features: int
    :param path_h5: The path to the directory which contains the HDF5 file.
    :type path_h5: str
    :param fname: The naming of the HDF5 file, without the .h5 appendix.
    :type fname: str
    :param group: The group from which we take data in the HDF5 file. Typically this is "events", "testpulses" or
        "noise".
    :type group: str
    :param sample_frequency: The sample frequency of the recording.
    :type sample_frequency: int
    :param record_length: The record length of the recording.
    :type record_length: int
    :param nmbr_channels: The number of channels in the HDF5 set.
    :type nmbr_channels: int
    :param datasets: This dictionary describes which datasets are loaded from the HDF5 set. The keys of the dict
        are the names that will be displayed in the VizTool. The values are lists, where the first element is the name
        of the data set in the HDF5 group, followed by two elements which are either an integer or None. If they are
        integer, they correspond to the index in the first and second dimension of the data set. If they are None, then
        no indexing is done in this dimension.
    :type datasets: dict
    :param table_names: The VizTool shows a table of the parameters of events that are currently selected. With this list,
        names (same names as are the keys in the datasets dict) can be provided to choose the parameters that are
        included in the table.
    :type table_names: list
    :param bins: The number of bins for the histograms that are displayed of the selected parameters.
    :type bins: int
    :param batch_size: The batch size for the calculation of standard events.
    :type batch_size: int
    """

    def __init__(self, csv_path=None, nmbr_features=None, path_h5=None, fname=None, group=None, sample_frequency=25000,
                 record_length=16384,
                 nmbr_channels=None, datasets=None, table_names=None, bins=100, batch_size=1000, *args, **kwargs):

        assert np.logical_xor(csv_path is None, path_h5 is None), 'Use either reading from CSV or from HDF5!'

        if csv_path is not None:
            self.mode = 'csv'
            assert np.logical_xor(csv_path is not None,
                                  nmbr_features is None), 'Opening a CSV file needs the argument nmbr_features!'
        elif path_h5 is not None:
            self.mode = 'h5'
            assert np.logical_xor(path_h5 is None,
                                  fname is not None and group is not None and datasets is not None and nmbr_channels is not None), 'Opening a H5 file needs the arguments fname, group, datasets and nmbr_channels!'
        else:
            raise KeyError('Put either CSV of H5 path!')

        # CSV
        if self.mode == 'csv':
            self.csv_path = csv_path
            self.nmbr_features = nmbr_features
            self.names = np.hstack(pd.read_csv(csv_path).values[:nmbr_features])
            self.data = pd.read_csv(self.csv_path, header=52, sep='\t', names=self.names)
            self.events_in_file = False

        # HDF5 set
        elif self.mode == 'h5':
            self.dh = DataHandler(nmbr_channels=nmbr_channels, sample_frequency=sample_frequency,
                                     record_length=record_length)
            self.dh.set_filepath(path_h5=path_h5, fname=fname, appendix=False)
            self.path_h5 = path_h5
            self.fname = fname
            try:
                with h5py.File(self.dh.path_h5, 'r') as f:
                    dummy = f[group]['event'][0, 0]
                    del dummy
                self.events_in_file = True
            except:
                self.events_in_file = False
            self.group = group
            self.nmbr_channels = nmbr_channels
            self.datasets = datasets  # this is a dictionary of dataset names, channel flags and feature indices
            self.data = {}
            self.names = []
            for k in datasets.keys():
                try:
                    if datasets[k][1] is not None:  # not a single channel flag
                        if datasets[k][2] is not None:
                            self.data[k] = self.dh.get(group, datasets[k][0])[datasets[k][1], :,
                                           datasets[k][2]]  # the indices
                        else:
                            self.data[k] = self.dh.get(group, datasets[k][0])[datasets[k][1]]
                    else:  # single channel flag
                        if datasets[k][2] is not None:
                            self.data[k] = self.dh.get(group, datasets[k][0])[:, datasets[k][2]]  # the indices
                        else:
                            self.data[k] = self.dh.get(group, datasets[k][0])[:]
                    self.names.append(k)
                except:
                    print(f'Could not include dataset {k}.')
            self.data = pd.DataFrame(self.data)

        # general
        self.data['Index'] = self.data.index
        if table_names is None:
            self.table_names = ['Index', self.names[1], self.names[2], self.names[3]]
        else:
            self.table_names = table_names
        self.N = len(self.data)
        self.bins = bins
        self.remaining_idx = np.array(list(self.data.index))
        self.color_flag = np.ones(len(self.remaining_idx))
        self.savepath = None
        self.batch_size = batch_size

    def set_idx(self, remaining_idx: list):
        """
        Display only the events with these indices.

        :param remaining_idx: A list of the indices that should be displayed.
        :type remaining_idx: list
        """
        assert len(remaining_idx.shape) == 1, 'remaining_idx needs to be a list of integers!'
        remaining_idx = np.array(remaining_idx)
        try:
            self.data = self.data.loc[remaining_idx]
            self.remaining_idx = remaining_idx
            if hasattr(self, 'f0'):
                self.f0.data[0].selectedpoints = list(range(len(self.data)))
                self.sel = list(range(len(self.data)))
                self._update_axes(self.xaxis, self.yaxis)
        except:
            raise NotImplementedError('You cannot use the set_idx function anymore once you applied cuts in this method!')

    def set_colors(self, color_flag: list):
        """
        Provide a list with numerical values, that correspond to the color intensities of the events.

        :param color_flag: The color intensities of the events.
        :type color_flag: list
        """
        assert len(self.remaining_idx) == len(color_flag), 'color flag must have same length as remaining indices!'
        self.color_flag = np.array(color_flag)

    def show(self):
        """
        Start the interactive visualization.
        """
        py.init_notebook_mode()

        # scatter plot
        self.f0 = go.FigureWidget([go.Scattergl(y=self.data[self.names[0]],
                                                x=self.data[self.names[0]],
                                                mode='markers',
                                                marker_color=self.color_flag)])
        scatter = self.f0.data[0]
        self.f0.layout.xaxis.title = self.names[0]
        self.f0.layout.yaxis.title = self.names[0]
        self.xaxis = self.names[0]
        self.yaxis = self.names[0]

        scatter.marker.opacity = 0.5

        self.sel = np.arange(len(self.remaining_idx))

        # histograms
        self.f1 = go.FigureWidget([go.Histogram(x=self.data[self.names[0]], nbinsx=self.bins)])
        self.f1.layout.xaxis.title = self.names[0]
        self.f1.layout.yaxis.title = 'Counts'

        self.f2 = go.FigureWidget([go.Histogram(x=self.data[self.names[0]], nbinsx=self.bins)])
        self.f2.layout.xaxis.title = self.names[0]
        self.f2.layout.yaxis.title = 'Counts'

        # dropdown menu
        axis_dropdowns = interactive(self._update_axes, yaxis=self.data.select_dtypes('float64').columns,
                                     xaxis=self.data.select_dtypes('float64').columns)

        # table
        self.t = go.FigureWidget([go.Table(
            header=dict(values=self.table_names,
                        fill=dict(color='#C2D4FF'),
                        align=['left'] * 5),
            cells=dict(values=[self.data[col] for col in self.table_names],
                       fill=dict(color='#F5F8FF'),
                       align=['left'] * 5))])

        scatter.on_selection(self._selection_scatter_fn)

        # button for drop
        cut_button = widgets.Button(description="Cut Selected")
        cut_button.on_click(self._button_cut_fn)

        # button for linear
        linear_button = widgets.Button(description="Linear")
        linear_button.on_click(self._button_linear_fn)

        # button for log
        log_button = widgets.Button(description="Log")
        log_button.on_click(self._button_log_fn)

        # button for export
        input_text = widgets.Text()
        input_text.on_submit(self._set_savepath)

        export_button = widgets.Button(description="Export Selected")
        self.output = widgets.Output()
        export_button.on_click(self._button_export_fn)

        # button for save
        if self.mode == 'h5' and self.events_in_file:
            save_button = widgets.Button(description="Save Selected")
            self.output = widgets.Output()
            save_button.on_click(self._button_save_fn)

        # plot event
        if self.mode == 'h5' and self.events_in_file:
            ev = self.dh.get(self.group, 'event')[:, 0, :]
            self.f3 = go.FigureWidget([go.Scattergl(
                x=np.arange(len(ev[0])),
                y=ev[c] - np.mean(ev[c, :500]),
                mode='lines',
                name='Channel {}'.format(c)
            ) for c in range(len(ev))])
            self.f3.layout.xaxis.title = 'Sample Index'
            self.f3.layout.yaxis.title = 'Amplitude (V)'
            self.f3.layout.title = 'Event idx {}'.format(0)

            scatter.on_click(self._plot_event)

            # slider

            self.slider = widgets.SelectionSlider(description='Event idx', options=self.remaining_idx[self.sel],
                                                  layout=Layout(width='500px'))
            self.slider_out = widgets.interactive(self._plot_event_slider, i=self.slider)

        # button for standard event
        if self.mode == 'h5' and self.events_in_file:
            sev_button = widgets.Button(description="Calc SEV")
            sev_button.on_click(self._button_sev_fn)

        # Put everything together
        if self.mode == 'csv':
            display(VBox((HBox(axis_dropdowns.children), self.f0,
                          HBox([cut_button, input_text, export_button]), self.output,
                          HBox([linear_button, log_button]),
                          self.f1, self.f2, self.t)))
        elif self.mode == 'h5' and self.events_in_file:
            display(VBox((HBox(axis_dropdowns.children), self.f0,
                          HBox([cut_button, input_text, export_button, save_button, sev_button]), self.output,
                          self.slider_out, self.f3,
                          HBox([linear_button, log_button]),
                          self.f1, self.f2, self.t)))
        else:
            raise NotImplementedError('Mode {} is not implemented!'.format(self.mode))

    # private

    def _update_axes(self, xaxis, yaxis):
        self.xaxis = xaxis
        self.yaxis = yaxis
        scatter = self.f0.data[0]
        scatter.x = self.data[xaxis]
        scatter.y = self.data[yaxis]
        histx = self.f1.data[0]
        histx.x = self.data[xaxis][self.remaining_idx[self.sel]]
        histy = self.f2.data[0]
        histy.x = self.data[yaxis][self.remaining_idx[self.sel]]
        self.f0.layout.xaxis.title = xaxis
        self.f0.layout.yaxis.title = yaxis
        self.f1.layout.xaxis.title = xaxis
        self.f2.layout.xaxis.title = yaxis

    def _selection_scatter_fn(self, trace, points, selector):
        self.t.data[0].cells.values = [self.data.loc[self.remaining_idx[points.point_inds]][col] for col in
                                       self.table_names]
        self.f1.data[0].x = self.data[self.xaxis][self.remaining_idx[points.point_inds]]
        self.f2.data[0].x = self.data[self.yaxis][self.remaining_idx[points.point_inds]]
        self.sel = points.point_inds
        self.slider.options = self.remaining_idx[self.sel]

    def _plot_event(self, trace, points, state):
        with h5py.File(self.path_h5 + self.fname + '.h5', 'r') as f:
            if len(points.point_inds) > 1:
                print('Click only one Event!')
            else:
                for i in self.remaining_idx[points.point_inds]:
                    ev = np.array(f[self.group]['event'][:, i, :])
                    for c in range(len(ev)):
                        self.f3.data[c].y = ev[c] - np.mean(ev[c, :500])
                    self.f3.update_layout(title='Event idx {}'.format(i))

    def _plot_event_slider(self, i):
        with h5py.File(self.path_h5 + self.fname + '.h5', 'r') as f:
            ev = np.array(f[self.group]['event'][:, i, :])
            for c in range(len(ev)):
                self.f3.data[c].y = ev[c] - np.mean(ev[c, :500])
            self.f3.update_layout(title='Event idx {}'.format(i))

    def _button_cut_fn(self, b):
        if self.sel is self.data.index:
            with self.output:
                print('Select events first!')
        else:
            self.data = self.data.drop(self.remaining_idx[self.sel])
            self.remaining_idx = np.delete(self.remaining_idx, self.sel)
            self.color_flag = np.delete(self.color_flag, self.sel)
            self.f0.data[0].selectedpoints = list(range(len(self.data)))
            self.sel = list(range(len(self.data)))
            self._update_axes(self.xaxis, self.yaxis)
            with self.output:
                print('Selected events removed from dataset.')

    def _button_log_fn(self, b):
        self.f1.update_yaxes(type="log")
        self.f2.update_yaxes(type="log")

    def _button_linear_fn(self, b):
        self.f1.update_yaxes(type="linear")
        self.f2.update_yaxes(type="linear")

    def _button_export_fn(self, b):
        if self.savepath is None:
            with self.output:
                print('Set savepath first!')
        else:
            np.savetxt(self.savepath + '.csv', self.remaining_idx[self.sel], delimiter=",")
            with self.output:
                print('Saved to ' + self.savepath + '.csv')

    def _button_save_fn(self, b):
        if self.savepath is None:
            with self.output:
                print('Set savepath first!')
        else:
            cut_flag = np.zeros(self.N, dtype=bool)
            cut_flag[self.remaining_idx[self.sel]] = True
            self.dh.apply_logical_cut(cut_flag=cut_flag,
                                      naming=self.savepath,
                                      channel=0,
                                      type=self.group,
                                      delete_old=True)
            with self.output:
                print('Saved as ' + self.savepath + ' in the HDF5 file.')

    def _button_sev_fn(self, b):
        if self.sel is self.data.index:
            with self.output:
                print('Select events first!')
        else:
            with h5py.File(self.path_h5 + self.fname + '.h5', 'r') as f:
                nmbr_batches = int(len(self.sel) / self.batch_size)
                self.sevs = [np.zeros(f[self.group]['event'][0, 0].shape[0]) for c in
                             range(self.nmbr_channels)]
                for b in range(nmbr_batches):
                    for c in range(self.nmbr_channels):
                        start = int(b * self.batch_size)
                        stop = int((b + 1) * self.batch_size)
                        ev = np.array(f[self.group]['event'][c, self.remaining_idx[self.sel[start:stop]]])
                        ev -= np.mean(ev[:, :500], axis=1, keepdims=True)
                        self.sevs[c] += np.sum(ev, axis=0)
                for c in range(self.nmbr_channels):
                    start = int(nmbr_batches * self.batch_size)
                    ev = np.array(f[self.group]['event'][c, self.remaining_idx[self.sel[start:]]])
                    ev -= np.mean(ev[:, :500], axis=1, keepdims=True)
                    self.sevs[c] += np.sum(ev, axis=0)
                    self.sevs[c] /= len(self.sel)
                with self.output:
                    print('Standardevent calculated.')

                for c in range(len(self.sevs)):
                    self.f3.data[c].y = self.sevs[c]
                self.f3.update_layout(title='Standard Event')

    def _set_savepath(self, path):
        self.savepath = path.value
        with self.output:
            print('Set savepath to ' + path.value)
