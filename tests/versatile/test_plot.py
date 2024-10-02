import pytest
import numpy as np
import cait.versatile as vai

x = np.linspace(0.1,1,100)

DATA = dict(
    line = dict(
        line1 = [ x , np.sin(x) ],
        line2 = [ x , np.cos(x) ]
    ),
    scatter = dict(
        scatter1 = [ x , x**2 ],
        scatter2 = [ x , -x ]
    ),
    histogram = dict(
        hist1 = [None, np.sin(x)],
        hist2 = [100, np.sin(x)],
        hist3 = [(0.1,1,20), np.sin(x)],
        hist4 = [np.arange(1,10), np.sin(x)] 
    ),
    axes = dict(
        xaxis = {"label": "xlabel", "scale": "log"},
        yaxis = {"label": "ylabel", "scale": "linear"}
    )
)

def test_not_implemented():
    with pytest.raises(NotImplementedError):
        vai.Viewer(backend="other_backend")

class TestViewerPlotly:
    BACKEND = "plotly"
    SHOW_CONTROLS = True
    TEMPLATE = "seaborn"

    def test_basic(self):
        vai.Viewer(backend=self.BACKEND, show_controls=self.SHOW_CONTROLS, 
                        template=self.TEMPLATE)
        vai.Viewer(data=DATA, backend=self.BACKEND, 
                        show_controls=self.SHOW_CONTROLS, template=self.TEMPLATE)
        vai.Viewer(data=DATA, backend=self.BACKEND, 
                        show_controls=self.SHOW_CONTROLS, template=self.TEMPLATE)

    def test_add_update(self):
        v = vai.Viewer(backend=self.BACKEND, 
                            show_controls=self.SHOW_CONTROLS, template=self.TEMPLATE)
        v.add_line(x=DATA["line"]["line1"][0], y=DATA["line"]["line1"][1], name="line")
        v.add_scatter(x=DATA["scatter"]["scatter1"][0], 
                      y=DATA["scatter"]["scatter1"][1], name="scatter")
        v.add_histogram(data=DATA["scatter"]["scatter1"][1], bins=10, name="histogram1")
        v.add_histogram(data=DATA["scatter"]["scatter2"][1], bins=(0,1,10), name="histogram2")

        v.update_line(name="line", x=DATA["line"]["line2"][0], y=DATA["line"]["line2"][1])
        v.update_scatter(name="scatter", x=DATA["scatter"]["scatter2"][0], 
                         y=DATA["scatter"]["scatter2"][1])
        v.update_histogram(name="histogram1", data=DATA["scatter"]["scatter2"][1], bins=10)
        v.update_histogram(name="histogram2", data=DATA["scatter"]["scatter1"][1], bins=(0,1,10))

    def test_getter_setter(self):
        v = vai.Viewer(backend=self.BACKEND, 
                            show_controls=self.SHOW_CONTROLS, template=self.TEMPLATE)
        v.get_figure()

        v.set_xlabel("x")
        v.set_ylabel("y")

        v.set_xscale("linear")
        v.set_xscale("log")
        v.set_yscale("linear")
        v.set_yscale("log")

    def test_legend(self):
        v = vai.Viewer(backend=self.BACKEND, show_controls=self.SHOW_CONTROLS,
                            template=self.TEMPLATE)
        v.show_legend(True)
        v.show_legend(False)

    def test_edit_artist(self):
        v = vai.Viewer(backend=self.BACKEND, 
                            show_controls=self.SHOW_CONTROLS, template=self.TEMPLATE)
        v.add_line(x=DATA["line"]["line1"][0], y=DATA["line"]["line1"][1], name="line")
        v.get_artist("line").line.dash = "dash"

class TestViewerMPL(TestViewerPlotly):
    BACKEND = "mpl"
    TEMPLATE = "seaborn-v0_8"

    def test_legend(self):
        v = vai.Viewer(backend=self.BACKEND, show_controls=self.SHOW_CONTROLS,
                            template=self.TEMPLATE)
        v.show_legend(True)

        with pytest.raises(NotImplementedError):
            v.show_legend(False)

    def test_edit_artist(self):
        v = vai.Viewer(backend=self.BACKEND, 
                            show_controls=self.SHOW_CONTROLS, template=self.TEMPLATE)
        v.add_line(x=DATA["line"]["line1"][0], y=DATA["line"]["line1"][1], name="line")
        v.get_artist("line").set_linestyle("--")
        v.update()

class TestViewerUniplot(TestViewerPlotly):
    BACKEND = "uniplot"
    SHOW_CONTROLS = False   # Otherwise the plot waits for stdin

    def test_legend(self):
        ... # Implementation for uniplot makes no sense

    def test_edit_artist(self):
        v = vai.Viewer(backend=self.BACKEND, 
                            show_controls=self.SHOW_CONTROLS, template=self.TEMPLATE)
        v.add_line(x=DATA["line"]["line1"][0], y=DATA["line"]["line1"][1], name="line")

        with pytest.raises(NotImplementedError):
            v.get_artist("line")

class TestLinePlotly:
    BACKEND = "plotly"
    SHOW_CONTROLS = True
    ALLOW_LOG = True
    TEMPLATE = "seaborn"

    def test_basic(self):
        vai.Line(y=DATA["line"]["line1"][1], 
                      backend=self.BACKEND, show_controls=self.SHOW_CONTROLS, 
                      template=self.TEMPLATE)
        vai.Line(x=DATA["line"]["line1"][0], 
                      y=DATA["line"]["line1"][1], 
                      backend=self.BACKEND, show_controls=self.SHOW_CONTROLS, 
                      template=self.TEMPLATE)
        vai.Line(y={name: data[1] for name, data in DATA["line"].items()},
                      backend=self.BACKEND, show_controls=self.SHOW_CONTROLS,
                      template=self.TEMPLATE)
        vai.Line(y={name: data[1] for name, data in DATA["line"].items()},
                      xscale="log" if self.ALLOW_LOG else "linear", 
                      yscale="log" if self.ALLOW_LOG else "linear", 
                      xlabel="xlabel", 
                      ylabel="ylabel",
                      backend=self.BACKEND, show_controls=self.SHOW_CONTROLS,
                      template=self.TEMPLATE)
        vai.Line(y=DATA["line"]["line1"][1], 
                      template=self.TEMPLATE, 
                      backend=self.BACKEND, show_controls=self.SHOW_CONTROLS,
                      xrange=(10,20),
                      yrange=(10,20))
        vai.Line([DATA["line"]["line1"][1], DATA["line"]["line1"][1]], 
                      backend=self.BACKEND, show_controls=self.SHOW_CONTROLS,
                      template=self.TEMPLATE)
        vai.Line({"line": [x, DATA["line"]["line1"][1]]}, 
                      backend=self.BACKEND, show_controls=self.SHOW_CONTROLS,
                      template=self.TEMPLATE)

class TestLineMPL(TestLinePlotly):
    BACKEND = "mpl"
    TEMPLATE = "seaborn-v0_8"

class TestLineUniplot(TestLinePlotly):
    BACKEND = "uniplot"
    SHOW_CONTROLS = False   # Otherwise the plot waits for stdin
    ALLOW_LOG = False       # Uniplot backend cannot handle 0 values with log-scale

class TestScatterPlotly:
    BACKEND = "plotly"
    SHOW_CONTROLS = True
    ALLOW_LOG = True
    TEMPLATE = "seaborn"

    def test_basic(self):
        vai.Scatter(y=DATA["scatter"]["scatter1"][1], backend=self.BACKEND, show_controls=self.SHOW_CONTROLS, template=self.TEMPLATE)
        vai.Scatter(x=DATA["scatter"]["scatter1"][0],
                         y=DATA["scatter"]["scatter1"][1], backend=self.BACKEND, show_controls=self.SHOW_CONTROLS, template=self.TEMPLATE)
        vai.Scatter(y={name: data[1] for name, data in DATA["scatter"].items()}, 
                         backend=self.BACKEND, show_controls=self.SHOW_CONTROLS, template=self.TEMPLATE)
        vai.Scatter(y={name: data[1] for name, data in DATA["scatter"].items()},
                         xscale="log" if self.ALLOW_LOG else "linear", 
                         yscale="log" if self.ALLOW_LOG else "linear", 
                         xlabel="xlabel", 
                         ylabel="ylabel", 
                         backend=self.BACKEND, show_controls=self.SHOW_CONTROLS, template=self.TEMPLATE)
        vai.Scatter(y=DATA["scatter"]["scatter1"][1], 
                         template=self.TEMPLATE,
                         backend=self.BACKEND, show_controls=self.SHOW_CONTROLS,
                         xrange=(10,20),
                         yrange=(10,20))
        vai.Scatter([DATA["line"]["line1"][1], DATA["line"]["line1"][1]],
                         backend=self.BACKEND, show_controls=self.SHOW_CONTROLS, template=self.TEMPLATE)
        vai.Scatter({"line": [x, DATA["line"]["line1"][1]]},
                         backend=self.BACKEND, show_controls=self.SHOW_CONTROLS, template=self.TEMPLATE)

class TestScatterMPL(TestScatterPlotly):
    BACKEND = "mpl"
    TEMPLATE = "seaborn-v0_8"

class TestScatterUniplot(TestScatterPlotly):
    BACKEND = "uniplot"
    SHOW_CONTROLS = False   # Otherwise the plot waits for stdin
    ALLOW_LOG = False       # Uniplot backend cannot handle 0 values with log-scale

class TestHistogramPlotly:
    BACKEND = "plotly"
    SHOW_CONTROLS = True
    TEMPLATE = "seaborn"
    ALLOW_LOG = True

    def test_basic(self):
        vai.Histogram(data=DATA["scatter"]["scatter1"][1], backend=self.BACKEND, show_controls=self.SHOW_CONTROLS, template=self.TEMPLATE)
        vai.Histogram(data=DATA["scatter"]["scatter1"][1], bins=10, 
                           backend=self.BACKEND, show_controls=self.SHOW_CONTROLS, template=self.TEMPLATE)
        vai.Histogram(data=DATA["scatter"]["scatter1"][1], bins=(0,1,10),
                           backend=self.BACKEND, show_controls=self.SHOW_CONTROLS, template=self.TEMPLATE)
        vai.Histogram(data=dict(first=DATA["scatter"]["scatter1"][1],
                                     second=DATA["scatter"]["scatter2"][1]), 
                           bins=(0,1,10), 
                           backend=self.BACKEND, show_controls=self.SHOW_CONTROLS, template=self.TEMPLATE)
        vai.Histogram(data=DATA["scatter"]["scatter1"][1], bins=10, 
                           template=self.TEMPLATE, backend=self.BACKEND, 
                           show_controls=self.SHOW_CONTROLS)
        vai.Histogram(data=DATA["scatter"]["scatter1"][1], 
                           xscale="log" if self.ALLOW_LOG else "linear", 
                           yscale="log" if self.ALLOW_LOG else "linear", 
                           xlabel="xlabel", 
                           ylabel="ylabel", 
                           backend=self.BACKEND, show_controls=self.SHOW_CONTROLS,
                           template=self.TEMPLATE,
                           xrange=(10,20),
                           yrange=(10,20))
        vai.Histogram([DATA["line"]["line1"][1], DATA["line"]["line1"][1]], 
                            backend=self.BACKEND, show_controls=self.SHOW_CONTROLS, template=self.TEMPLATE)
        
        with pytest.raises(TypeError):
            vai.Histogram(data=DATA["scatter"]["scatter1"][1], bins="nonsense", 
                               backend=self.BACKEND, show_controls=self.SHOW_CONTROLS, template=self.TEMPLATE)
        with pytest.raises(TypeError):
            vai.Histogram(data=DATA["scatter"]["scatter1"][1], bins=(1,2), 
                               backend=self.BACKEND, show_controls=self.SHOW_CONTROLS, template=self.TEMPLATE)

class TestHistogramMPL(TestHistogramPlotly):
    BACKEND = "mpl"
    TEMPLATE = "seaborn-v0_8"

class TestHistogramUniplot(TestHistogramPlotly):
    BACKEND = "uniplot"
    SHOW_CONTROLS = False   # Otherwise the plot waits for stdin
    ALLOW_LOG = False       # Uniplot backend cannot handle 0 values with log-scale

class TestPreview:
    BACKEND = "plotly"
    SHOW_CONTROLS = True 

    ...