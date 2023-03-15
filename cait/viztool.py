import numpy as np

import plotly.graph_objs as go
import plotly.offline as py
import pandas as pd  # new to cait
import numpy as np
import ipywidgets as widgets  # new to cait
from ipywidgets import interactive, HBox, VBox, Button, Layout
import h5py
from IPython.display import display
import datashader as ds
import plotly.express as px

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
    :param datahandler: A DataHandler instance to be used instead of specifying the location of the HDF5 file and sample_frequency, channels, ...
    :type datahandler: cait.DataHandler
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
        For quick inspection, the values can also be 1d numpy arrays. For reproducibility, it is nevertheless recommended to 
        include such additional datasets in the HDF5 file first (using the dh.include_values() method) and loading it from there.
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

    def __init__(self, csv_path=None, nmbr_features=None, datahandler=None, path_h5=None, fname=None, group=None, sample_frequency=25000,
                 record_length=16384,
                 nmbr_channels=None, datasets=None, table_names=None, bins=100, batch_size=1000, *args, **kwargs):

        # exactly one of the options has to be chosen
        assert np.sum([csv_path is not None, path_h5 is not None, datahandler is not None])==1, 'Use either reading from csv OR from HDF5 OR from existing datahandler!'

        if datahandler is not None:
            assert isinstance(datahandler, DataHandler), 'datahandler must be a DataHandler instance.'
            self.mode = 'h5'
            assert np.all([path_h5 is None, 
                           fname is None, 
                           nmbr_channels is None,
                           sample_frequency==25000,
                           record_length==16384]), 'You cannot set path_h5, fname, nmbr_channels, sample_frequency or record_length if datahandler is provided.'
        elif csv_path is not None:
            self.mode = 'csv'
            assert np.logical_xor(csv_path is not None,
                                  nmbr_features is None), 'Opening a CSV file needs the argument nmbr_features!'
        elif path_h5 is not None:
            self.mode = 'h5'
            assert np.logical_xor(path_h5 is None,
                                  fname is not None and group is not None and datasets is not None and nmbr_channels is not None), 'Opening a H5 file needs the arguments fname, group, datasets and nmbr_channels!'
        else:
            raise KeyError('Provide datahandler, csv or h5 path!')

        # CSV
        if self.mode == 'csv':
            self.csv_path = csv_path
            self.nmbr_features = nmbr_features
            self.names = np.hstack(pd.read_csv(csv_path).values[:nmbr_features])
            self.data = pd.read_csv(self.csv_path, header=52, sep='\t', names=self.names)
            self.events_in_file = False

        # HDF5 set
        elif self.mode == 'h5':
            if datahandler is not None:
                self.dh = datahandler
                self.path_h5 = datahandler.path_h5
            else:
                self.dh = DataHandler(nmbr_channels=nmbr_channels, sample_frequency=sample_frequency,
                                     record_length=record_length)
                self.dh.set_filepath(path_h5=path_h5, fname=fname, appendix=False)
                self.path_h5 = path_h5 + fname + '.h5'
            try:
                with h5py.File(self.dh.path_h5, 'r') as f:
                    dummy = f[group]['event'][0, 0]
                    del dummy
                self.events_in_file = True
            except:
                self.events_in_file = False
            self.group = group

            self.datasets = datasets  # this is a dictionary of dataset names, channel flags and feature indices
            self.data = {}
            self.names = []
            for k in datasets.keys():
                try:
                    # 1d numpy arrays can be included directly.
                    if isinstance(datasets[k],np.ndarray):
                        if datasets[k].ndim == 1:
                            try:
                                self.data[k] = np.asarray(datasets[k], dtype='float64')
                            except:
                                print(f'Failed to include {k} because it could not be converted to float64.')
                        else:
                            print(f'Failed to include {k} because it is not one-dimensional.')
                        
                    if isinstance(datasets[k],list):
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
                #self.f0.data[0].selectedpoints = list(range(len(self.data)))
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

        # Initialize plots (empty yet)
        self.f0 = go.FigureWidget([go.Scatter(  name='Datapoints',
                                                mode='markers',
                                                marker=dict(
                                                        size=6,
                                                        opacity=0.8,
                                                        colorscale='plasma'
                                                    ),
                                                xaxis='x',
                                                yaxis='y',
                                                visible=False),
                                    go.Heatmap( name='Density',
                                                xaxis='x',
                                                yaxis='y',
                                                colorscale='jet',
                                                colorbar={'thickness':5, 'outlinewidth':0},
                                                visible=False),
                                  go.Histogram( name='Histogram y',
                                                nbinsx=self.bins,
                                                xaxis='x2',
                                                marker=dict(
                                                    color = 'rgba(94,94,94,1)'
                                                    )
                                                ),
                                  go.Histogram( name='Histogram x',
                                                nbinsx=self.bins,
                                                yaxis='y2',
                                                marker=dict(
                                                    color = 'rgba(94,94,94,1)'
                                                )
                                            )])
        
        self.f0.update_layout(
                        autosize = True,
                        xaxis = dict(
                            zeroline = False,
                            domain = [0,0.84],
                            showgrid = True
                        ),
                        yaxis = dict(
                            zeroline = False,
                            domain = [0,0.84],
                            showgrid = True
                        ),
                        xaxis2 = dict(
                            zeroline = False,
                            domain = [0.85,1],
                            showgrid = True
                        ),
                        yaxis2 = dict(
                            zeroline = False,
                            domain = [0.85,1],
                            showgrid = True
                        ),
                        xaxis3 = dict(
                            zeroline = False,
                            domain = [0.85,1],
                            showgrid = True,
                            showticklabels = False
                        ),
                        yaxis3 = dict(
                            zeroline = False,
                            domain = [0.85,1],
                            showgrid = True,
                            showticklabels = False
                        ),
                        height = 800,
                        #width = 800,
                        bargap = 0,
                        hovermode = 'closest',
                        showlegend = False,
                        template='ggplot2'
                    )
        
        scatter = self.f0.data[0]

        self.f0.layout.xaxis.title = self.names[0]
        self.f0.layout.yaxis.title = self.names[0]

        # data point selection
        self.sel = np.arange(len(self.remaining_idx))
        scatter.on_selection(self._selection_scatter_fn)
        scatter.on_deselect(self._deselection_scatter_fn)

        # DROPDOWN SETTINGS
        # Initializing dropdown values
        self.xaxis, self.yaxis, self.which, self.color, self.scale = None, None, None, 'None', 'linear'
        # Creating dropdown menues
        axis_dropdowns = interactive(self._update_axes, 
                                     y=self.data.select_dtypes('float64').columns,
                                     x=self.data.select_dtypes('float64').columns, 
                                     color= ['None'] + self.data.select_dtypes('float64').columns.to_list(),
                                     which=['select','datapoints','density'],
                                     scale=['linear','log'])

        # table
        # self.t = go.FigureWidget([go.Table(
        #     header=dict(values=self.table_names,
        #                 fill=dict(color='#C2D4FF'),
        #                 align=['left'] * 5),
        #     cells=dict(values=[self.data[col] for col in self.table_names],
        #                fill=dict(color='#F5F8FF'),
        #                align=['left'] * 5))])

        # button for drop
        cut_button = widgets.Button(description="Cut Selected")
        cut_button.on_click(self._button_cut_fn)

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
            with h5py.File(self.path_h5, 'r') as f:
                ev = np.array(f[self.group]['event'][:,0,:])
                colors = px.colors.qualitative.Plotly
            traces = [go.Scatter(x=self.dh.record_window(ms=True),
                                 visible=False,
                                 mode='lines',
                                 name='Channel {}'.format(c),
                                 xaxis = 'x3',
                                 yaxis = 'y3',
                                 marker = {'color': colors[c%10]}
                                ) for c in range(len(ev))]
            
            # Make separate plot for events
            self.f1 = go.FigureWidget(traces)
            self.f1.update_layout(  xaxis3=dict(title="Time (ms)"),
                                    yaxis3=dict(title='Amplitude (V)'),
                                    legend=dict(x=0.99,
                                                y=0.95,
                                                xanchor="right",
                                                yanchor="top",
                                            ),
                                    template='ggplot2',
                                    legend_title_text='Event idx {}'.format(0)
                                    )
            self.f1.update_traces(dict(visible=True)) #default invisible for f0 plot but not for f1

            # Also add to figure0 for quick inspection 
            self.f0.add_traces(traces)

            # Plot upon event selection in scatter plot
            scatter.on_click(self._plot_event)

            # slider
            self.slider = widgets.SelectionSlider(description='Event idx', options=self.remaining_idx[self.sel],
                                                  layout=Layout(width='500px')
                                                  )
            self.slider_out = widgets.interactive(self._plot_event_slider, i=self.slider)

        # button for standard event
        if self.mode == 'h5' and self.events_in_file:
            sev_button = widgets.Button(description="Calc SEV")
            sev_button.on_click(self._button_sev_fn)

        # Put everything together
        if self.mode == 'csv':
            display(VBox((HBox(axis_dropdowns.children, layout=Layout(flex_flow='row wrap')), self.f0,
                          HBox([cut_button, input_text, export_button]), self.output)))  # , self.t
        elif self.mode == 'h5' and self.events_in_file:
            display(VBox((HBox(axis_dropdowns.children, layout=Layout(flex_flow='row wrap')), self.f0,
                          HBox([cut_button, input_text, export_button, save_button, sev_button]), self.output,
                          self.f1, self.slider_out)))  # , self.t
        elif self.mode == 'h5' and not self.events_in_file:
            display(VBox((HBox(axis_dropdowns.children, layout=Layout(flex_flow='row wrap')), self.f0,
                          HBox([cut_button, input_text, export_button]), self.output)))
        else:
            raise NotImplementedError('Mode {} is not implemented!'.format(self.mode))

    # private

    def _update_axes(self, which=None, x=None, y=None, color=None, scale=None):
        # check which property changed (only updated as much as necessary)
        which_changed = False if which is self.which else True
        x_changed = False if x is self.xaxis else True
        y_changed = False if y is self.yaxis else True
        color_changed = False if color is self.color else True
        scale_changed = False if scale is self.scale else True

        if which is not None: self.which = which
        if x is not None: self.xaxis = x
        if y is not None: self.yaxis = y
        if color is not None: self.color = color
        if scale is not None: self.scale = scale

        # setting x-lim and y-lim
        if self.xaxis is not None and (x_changed or which_changed):
            xlims = [np.min(self.data[self.xaxis]), np.max(self.data[self.xaxis])]
            self.f0.update_layout(xaxis_range=[1.02*xlims[0]-0.02*xlims[1], 1.02*xlims[1]-0.02*xlims[0]])
        if self.yaxis is not None and (y_changed or which_changed):
            ylims = [np.min(self.data[self.yaxis]), np.max(self.data[self.yaxis])]
            self.f0.update_layout(yaxis_range=[1.02*ylims[0]-0.02*ylims[1], 1.02*ylims[1]-0.02*ylims[0]])

        if self.which=='datapoints':
            if not self.f0.data[0].visible:
                self.f0.data[1].update({"visible":False})
                self.f0.data[0].update({"visible":True})
            if color_changed or which_changed:
                if self.color not in [None, 'None']:
                    self.f0.data[0].marker['color'] = self.data[self.color]
                    self.f0.data[0].marker['colorbar'] = {'thickness':5, 'outlinewidth':0}
                else:
                    self.f0.data[0].marker['color'] = self.color_flag

            # update scatter plot
            if self.xaxis is not None and (x_changed or which_changed): self.f0.data[0].x = self.data[self.xaxis]
            if self.yaxis is not None and (y_changed or which_changed): self.f0.data[0].y = self.data[self.yaxis]

        elif self.which=='density':
            if not self.f0.data[1].visible:
                self.f0.data[0].update({"visible":False})
                self.f0.data[1].update({"visible":True})

            # prepare density plot
            if self.xaxis is not None and self.yaxis is not None:
                cvs = ds.Canvas(plot_width=self.bins, plot_height=self.bins)
                agg = cvs.points(self.data, self.xaxis, self.yaxis)
                zero_mask = agg.values == 0
                if self.scale == 'log':
                    agg.values = np.log10(agg.values, where=np.logical_not(zero_mask))
                    self.f0.data[1].colorbar["tickprefix"] = '1.e'
                else:
                    self.f0.data[1].colorbar["tickprefix"] = None

                self.f0.data[1].colorbar["title"] = "counts"

                # json doesn't support np.nan which is why we have to convert to object dtype
                agg.values = np.asarray(agg.values, dtype=object) 
                agg.values[zero_mask] = None
                figim = px.imshow(agg, origin='lower')

                # update density plot
                if x_changed or which_changed: self.f0.data[1].update({"x": figim.data[0].x})
                if y_changed or which_changed: self.f0.data[1].update({"y": figim.data[0].y})
                self.f0.data[1].update({"z": figim.data[0].z})
            
        # update histograms
        if self.yaxis is not None and y_changed: self.f0.data[2].y = self.data[self.yaxis][self.remaining_idx[self.sel]]
        if self.xaxis is not None and x_changed: self.f0.data[3].x = self.data[self.xaxis][self.remaining_idx[self.sel]]
        if scale_changed:
            if self.scale is not None and self.f0.layout.xaxis2.type != self.scale:
                self.f0.layout.xaxis2.update(type=self.scale)
                self.f0.layout.yaxis2.update(type=self.scale)

        # update axis titles
        if x_changed: self.f0.layout.xaxis.title = self.xaxis
        if y_changed: self.f0.layout.yaxis.title = self.yaxis

    def _selection_scatter_fn(self, trace, points, selector):
        # self.t.data[0].cells.values = [self.data.loc[self.remaining_idx[points.point_inds]][col] for col in
        #                                self.table_names]
        if self.f0.data[0].visible:
            self.f0.data[2].y = self.data[self.yaxis][self.remaining_idx[points.point_inds]]
            self.f0.data[3].x = self.data[self.xaxis][self.remaining_idx[points.point_inds]]
            self.sel = points.point_inds
            self.slider.options = self.remaining_idx[self.sel]
        elif self.f0.data[1].visible:
            with self.output:
                raise NotImplementedError('plotly does not support selection of heatmaps at the moment. Therefore, selection is only possible in "datapoints" mode.')

    def _deselection_scatter_fn(self, trace, points):
        if self.f0.data[0].visible:
            self.f0.data[2].y = self.data[self.yaxis][self.remaining_idx]
            self.f0.data[3].x = self.data[self.xaxis][self.remaining_idx]
            self.sel = np.arange(len(self.remaining_idx)) # vizTool will treat all events as selected
            self.slider.options = self.remaining_idx

    def _plot_event(self, trace, points, state):
        with h5py.File(self.path_h5, 'r') as f:
            if len(points.point_inds) > 1:
                print('Click only one Event!')
            else:
                for i in self.remaining_idx[points.point_inds]:
                    ev = np.array(f[self.group]['event'][:, i, :])
                    n = len(ev)
                    for c in range(n):
                        self.f0.data[-n+c].y = ev[c] - np.mean(ev[c, :500])
                        if not self.f0.data[-n+c].visible: self.f0.data[-n+c].update({"visible":True})
                        self.f1.data[c].y = ev[c] - np.mean(ev[c, :500])
                    self.f1.update_layout(legend_title_text='Event idx {}'.format(i))

    def _plot_event_slider(self, i):
        with h5py.File(self.path_h5, 'r') as f:
            ev = np.array(f[self.group]['event'][:, i, :])
            for c in range(len(ev)):
                self.f1.data[c].y = ev[c] - np.mean(ev[c, :500])
            self.f1.update_layout(legend_title_text='Event idx {}'.format(i))

    def _button_cut_fn(self, b):
        if list(self.sel) == list(self.data.index):
            with self.output:
                print('Select subset of events first!')
        else:
            self.data = self.data.drop(self.remaining_idx[self.sel])
            self.remaining_idx = np.delete(self.remaining_idx, self.sel)
            self.color_flag = np.delete(self.color_flag, self.sel)
            #self.f0.data[0].selectedpoints = list(range(len(self.data)))
            self.sel = list(range(len(self.data)))
            self._update_axes(x=self.xaxis, y=self.yaxis, color=self.color)
            with self.output:
                print('Selected events removed from dataset.')

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
            with h5py.File(self.path_h5, 'r') as f:
                nmbr_batches = int(len(self.sel) / self.batch_size)
                self.sevs = [np.zeros(f[self.group]['event'][0, 0].shape[0]) for c in
                             range(self.dh.nmbr_channels)]
                for b in range(nmbr_batches):
                    for c in range(self.dh.nmbr_channels):
                        start = int(b * self.batch_size)
                        stop = int((b + 1) * self.batch_size)
                        ev = np.array(f[self.group]['event'][c, self.remaining_idx[self.sel[start:stop]]])
                        ev -= np.mean(ev[:, :500], axis=1, keepdims=True)
                        self.sevs[c] += np.sum(ev, axis=0)
                for c in range(self.dh.nmbr_channels):
                    start = int(nmbr_batches * self.batch_size)
                    ev = np.array(f[self.group]['event'][c, self.remaining_idx[self.sel[start:]]])
                    ev -= np.mean(ev[:, :500], axis=1, keepdims=True)
                    self.sevs[c] += np.sum(ev, axis=0)
                    self.sevs[c] /= len(self.sel)
                with self.output:
                    print('Standardevent calculated.')

                for c in range(len(self.sevs)):
                    self.f1.data[c].y = self.sevs[c]
                self.f1.update_layout(legend_title_text='Standard Event')

    def _set_savepath(self, path):
        self.savepath = path.value
        with self.output:
            print('Set savepath to ' + path.value)
