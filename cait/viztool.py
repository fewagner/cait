import numpy as np
import pandas as pd

import plotly.graph_objs as go
import plotly.colors as colors

import ipywidgets as widgets
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
                 nmbr_channels=None, datasets=None, table_names=None, bins=200, batch_size=1000, *args, **kwargs):

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
                self.dh.get(group, "event", 0, 0)
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

    def show(self, template='ggplot2'):
        """
        Start the interactive visualization.

        :param template: The plotly template to be used, e.g. "plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white", "none"
        :type template: str
        """
        # Initialize plots (empty yet)
        self.f0 = go.FigureWidget([go.Scatter(  name='Datapoints',
                                                mode='markers',
                                                marker=dict(
                                                        size=6,
                                                        opacity=0.8,
                                                        colorscale='plasma',
                                                        #line={'width':1}
                                                    ),
                                                xaxis='x',
                                                yaxis='y',
                                                visible=False),
                                    go.Heatmap( name='Density',
                                                xaxis='x',
                                                yaxis='y',
                                                hoverongaps=False,
                                                hovertemplate = '(%{x}, %{y})<br>counts: %{customdata}',
                                                colorscale='jet',
                                                colorbar={'thickness':5, 'outlinewidth':0},
                                                visible=False),
                                  go.Histogram( name='Histogram y',
                                                #nbinsy=self.bins,
                                                xaxis='x2',
                                                marker=dict(
                                                    color = 'rgba(94,94,94,1)'
                                                    )
                                                ),
                                  go.Histogram( name='Histogram x',
                                                #nbinsx=self.bins,
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
                        margin=dict(l=0, r=0, t=40, b=80),
                        hovermode = 'closest',
                        xaxis_hoverformat='0.3g',
                        yaxis_hoverformat='0.3g',
                        showlegend = False,
                        template=template,
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
        # Creating dropdown menus
        names = self.data.select_dtypes('float64').columns
        controls = widgets.interactive(self._update_axes, 
                    x=widgets.Dropdown(options=names,
                                       value=names[0],
                                       layout={'width': '20ex'},
                                       style={'description_width': '2ex'}), 
                    y=widgets.Dropdown(options=names,
                                       value=names[1],
                                       layout={'width': '20ex'},
                                       style={'description_width': '2ex'}
                                       ),
                    color=widgets.Dropdown(options=['None'] + names.to_list(),
                                           value='None',
                                           layout={'width': '26ex'},
                                           style={'description_width': '6ex'}
                                           ),
                    which=widgets.RadioButtons(options=['datapoints','density'],
                                               value='datapoints',
                                               layout={'width': '15ex'}
                                               ),
                    scale=widgets.RadioButtons(options=['linear','log'],
                                               value='linear',
                                               layout={'width': '11ex'}
                                               ))
        # Remove 'which' and 'scale' description from controls
        controls.children[3].description = ''
        controls.children[4].description = ''

        self.info_box = widgets.Output()
        with self.info_box: 
            self.info_box.clear_output()
            print('0 selected')
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

        if self.mode == 'h5' and self.events_in_file:
            # button for save
            save_button = widgets.Button(description="Save Selected")
            self.output = widgets.Output()
            save_button.on_click(self._button_save_fn)

            # button for standard event
            sev_button = widgets.Button(description="Calc SEV") 
            sev_button.on_click(self._button_sev_fn)
            
            # traces for event plot
            ev = self.dh.get(self.group, "event", None, 0, None)
            
            traces = [go.Scatter(x=self.dh.record_window(ms=True),
                                 visible=False,
                                 mode='lines',
                                 name='Channel {}'.format(c),
                                 xaxis='x3',
                                 yaxis='y3',
                                 marker={'color': colors.qualitative.Plotly[c%10]},
                                 showlegend=True
                                ) for c in range(len(ev))]
            
            # Add to figure0 as is for quick inspection 
            self.f0.add_traces(traces)

            # MAIN PARAMETER VIEW (currently not in use)
            #self.mp_button = widgets.ToggleButton(value=False, description="Show/Hide MP")
            #self.mp_button.observe(self._toggle_mp_event, "value")
            #mp_markers = [go.Scatter(   visible=False,
            #                            mode='markers',
            #                            name='MPs channel {}'.format(c),
            #                            xaxis='x3',
            #                            yaxis='y3',
            #                            marker={'color': colors[c%10], 'size': 6, 'line' : {'width' :2}},
            #                            showlegend=False
            #                            ) for c in range(len(ev))]
            #traces.extend(mp_markers)
            
            # Make separate plot for events
            self.f1 = go.FigureWidget(traces)
            self.f1.update_layout(  xaxis3=dict(title="Time (ms)"),
                                    yaxis3=dict(title='Amplitude (V)'),
                                    legend=dict(x=0.99,
                                                y=0.95,
                                                xanchor="right",
                                                yanchor="top",
                                            ),
                                    template=template,
                                    margin=dict(l=0, r=0, t=40, b=80),
                                    legend_title_text='Event idx {}'.format(0),
                                    hovermode="x unified",
                                    xaxis3_hoverformat='0.3g',
                                    yaxis3_hoverformat='0.3g'
                                    )
            self.f1.update_traces(dict(visible=True),
                                   lambda x: x.name.startswith("Channel")
                                   ) #default invisible for f0 plot but not for f1

            # Plot upon event selection in scatter plot
            scatter.on_click(self._plot_event)

            # slider
            self.slider = widgets.SelectionSlider(description='Event idx', options=self.remaining_idx[self.sel],
                                                  layout=widgets.Layout(width='500px')
                                                  )
            self.slider_out = widgets.interactive(self._plot_event_slider, i=self.slider)

        # Put everything together
        if self.mode == 'csv':
            display(widgets.VBox((widgets.HBox(controls.children, 
                               layout=widgets.Layout(flex_flow='row wrap',  
                                             justify_content='space-between',
                                             align_items='center')
                               ), 
                          self.f0,
                          widgets.HBox([self.info_box, cut_button, input_text, export_button],
                               layout=widgets.Layout(align_items='center')
                               ), 
                          self.output)
                          )
                    )  # , self.t
        elif self.mode == 'h5' and self.events_in_file:
            display(widgets.VBox((widgets.HBox(controls.children, 
                               layout=widgets.Layout(flex_flow='row wrap',
                                             justify_content='space-between',
                                             align_items='center')
                               ), 
                          self.f0,
                          widgets.HBox([self.info_box, cut_button, input_text, export_button, save_button, sev_button],
                               layout=widgets.Layout(align_items='center')
                               ), 
                          self.output,
                          self.f1, 
                          widgets.HBox([self.slider_out, 
                                         #self.mp_button
                                         ])
                            )
                            ))  # , self.t
        elif self.mode == 'h5' and not self.events_in_file:
            display(widgets.VBox((widgets.HBox(controls.children, 
                               layout=widgets.Layout(flex_flow='row wrap',
                                             justify_content='space-between',
                                             align_items='center')
                               ), 
                          self.f0,
                          widgets.HBox([self.info_box, cut_button, input_text, export_button],
                               layout=widgets.Layout(align_items='center')
                               ), 
                          self.output)
                        )
                    )
        else:
            raise NotImplementedError('Mode {} is not implemented!'.format(self.mode))

    # private

    def _update_axes(self, x=None, y=None, color=None, which=None, scale=None):
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
                    self.f0.data[0].marker.showscale = True
                    self.f0.data[0].marker.colorbar.title = self.color
                    self.f0.data[0].marker.colorbar.titleside = "right"
                else:
                    self.f0.data[0].marker['color'] = self.color_flag
                    self.f0.data[0].marker.showscale = False

            # update scatter plot
            if self.xaxis is not None and (x_changed or which_changed): self.f0.data[0].x = self.data[self.xaxis]
            if self.yaxis is not None and (y_changed or which_changed): self.f0.data[0].y = self.data[self.yaxis]

        elif self.which=='density':
            if not self.f0.data[1].visible:
                self.f0.data[0].update({"visible":False})
                self.f0.data[1].update({"visible":True})

            # prepare density plot
            if self.xaxis is not None and self.yaxis is not None:
                # Do binning with numpy
                # y first argument, so that we don't have to transpose output
                counts, y_edges, x_edges = np.histogram2d(self.data[self.yaxis], 
                                                     self.data[self.xaxis], 
                                                     self.bins) 
                x_centers = (x_edges[:-1] + x_edges[1:])/2
                y_centers = (y_edges[:-1] + y_edges[1:])/2
                mask = counts == 0

                if self.scale == 'log':
                    z = np.log10(counts, where=~mask)
                    self.f0.data[1].colorbar["tickprefix"] = '1.e'
                else:
                    z = counts
                    self.f0.data[1].colorbar["tickprefix"] = None

                self.f0.data[1].colorbar.title = "counts"
                self.f0.data[1].colorbar.titleside = "right"

                # json doesn't support np.nan which is why we have to convert to object dtype
                z = np.asarray(z, dtype=object)
                z[mask] = None

                # Independent of whether we display log values or not, we want the
                # count values to show. The required hovertemplate is specified above
                # (where Heatmap is constructed)
                self.f0.data[1].update({"x": x_centers, 
                                        "y": y_centers, 
                                        "z": z,
                                        "customdata": counts})


                # ALTERNATIVE IMPLEMENTATION (requires datashader, might be faster
                # for large amounts of data)
                # cvs = ds.Canvas(plot_width=self.bins, plot_height=self.bins)
                # agg = cvs.points(self.data, self.xaxis, self.yaxis)
                # zero_mask = agg.values == 0
                # if self.scale == 'log':
                #     agg.values = np.log10(agg.values, where=np.logical_not(zero_mask))
                #     self.f0.data[1].colorbar["tickprefix"] = '1.e'
                # else:
                #     self.f0.data[1].colorbar["tickprefix"] = None

                # self.f0.data[1].colorbar.title = "counts"
                # self.f0.data[1].colorbar.titleside = "right"

                # # json doesn't support np.nan which is why we have to convert to object dtype
                # agg.values = np.asarray(agg.values, dtype=object) 
                # agg.values[zero_mask] = None
                # figim = px.imshow(agg, origin='lower')

                # # update density plot
                # self.f0.data[1].update({"x": figim.data[0].x, "y": figim.data[0].y, "z": figim.data[0].z})
            
        # update histograms
        if self.yaxis is not None and y_changed: 
            temp = self.data[self.yaxis][self.remaining_idx[self.sel]]
            self.f0.data[2].update(
                    dict(y = temp,
                         ybins = {"start": np.min(temp), 
                                  "end": np.max(temp), 
                                  "size": (np.max(temp)-np.min(temp))/self.bins}))
        if self.xaxis is not None and x_changed: 
            temp = self.data[self.xaxis][self.remaining_idx[self.sel]]
            self.f0.data[3].update(
                    dict(x = temp,
                         xbins = {"start": np.min(temp), 
                                  "end": np.max(temp), 
                                  "size": (np.max(temp)-np.min(temp))/self.bins}))

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
            # USING THE FOLLOWING LINES, THE HISTOGRAMS WOULD BE RESAMPLED WITH THE
            # SELECTED DATAPOINTS
            #self.f0.data[2].y = self.data[self.yaxis][self.remaining_idx[points.point_inds]]
            #self.f0.data[3].x = self.data[self.xaxis][self.remaining_idx[points.point_inds]]
            self.sel = points.point_inds
            self.slider.options = self.remaining_idx[self.sel]

            with self.info_box:
                self.info_box.clear_output()
                print(f"{len(self.f0.data[0].selectedpoints)} selected")

        elif self.f0.data[1].visible:
            with self.output:
                raise NotImplementedError('plotly does not support selection of heatmaps at the moment. Therefore, selection is only possible in "datapoints" mode.')

    def _deselection_scatter_fn(self, trace, points):
        if self.f0.data[0].visible:
            # USING THE FOLLOWING LINES, THE HISTOGRAMS WOULD BE RESAMPLED WITH THE
            # SELECTED DATAPOINTS
            #self.f0.data[2].y = self.data[self.yaxis][self.remaining_idx]
            #self.f0.data[3].x = self.data[self.xaxis][self.remaining_idx]
            self.sel = np.arange(len(self.remaining_idx)) # vizTool will treat all events as selected
            self.slider.options = self.remaining_idx

            with self.info_box:
                self.info_box.clear_output()
                print('0 selected')

    def _plot_event(self, trace, points, state):
        if len(points.point_inds) > 1:
            print('Click only one Event!')
        else:
            for i in self.remaining_idx[points.point_inds]:
                ev = self.dh.get(self.group, "event", None, i, None)
                #if self.mp_button.value: mp = np.array(f[self.group]['mainpar'][:, i, 1:7], dtype=int)
                n = len(ev)
                for c in range(n):
                    self.f0.data[-n+c].y = ev[c] - np.mean(ev[c, :500])
                    if not self.f0.data[-n+c].visible: self.f0.data[-n+c].update({"visible":True})
                    self.f1.data[c].y = ev[c] - np.mean(ev[c, :500])
                    #if self.mp_button.value:
                    #    self.f1.data[n+c].x = self.f1.data[c].x[mp[c]]
                    #    self.f1.data[n+c].y = self.f1.data[c].y[mp[c]]
                self.f1.update_layout(legend_title_text='Event idx {}'.format(i))

    def _plot_event_slider(self, i):
        ev = self.dh.get(self.group, "event", None, i, None)
        #if self.mp_button.value: mp = np.array(f[self.group]['mainpar'][:, i, 1:7], dtype=int)
        n = len(ev)
        for c in range(n):
            self.f1.data[c].y = ev[c] - np.mean(ev[c, :500])
            #if self.mp_button.value:
            #    self.f1.data[n+c].x = self.f1.data[c].x[mp[c]]
            #    self.f1.data[n+c].y = self.f1.data[c].y[mp[c]]
        self.f1.update_layout(legend_title_text='Event idx {}'.format(i))

    # def _toggle_mp_event(self, b):
    #     shown = self.mp_button.value
    #     if shown: self._plot_event_slider(self.slider.value)

    #     self.f1.update_traces(dict(visible=shown), lambda x: x.name.startswith("MP"))            
    
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
            nmbr_batches = int(len(self.sel) / self.batch_size)
            self.sevs = [np.zeros(self.dh.get(self.group, "event", 0, 0).shape[0]) for c in range(self.dh.nmbr_channels)]

            for b in range(nmbr_batches):
                for c in range(self.dh.nmbr_channels):
                    start = int(b * self.batch_size)
                    stop = int((b + 1) * self.batch_size)
                    ev = self.dh.get(self.group, "event", c, self.remaining_idx[self.sel[start:stop]])
                    ev -= np.mean(ev[:, :500], axis=1, keepdims=True)
                    self.sevs[c] += np.sum(ev, axis=0)
            for c in range(self.dh.nmbr_channels):
                start = int(nmbr_batches * self.batch_size)
                ev = self.dh.get(self.group, "event", c, self.remaining_idx[self.sel[start:]])
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
