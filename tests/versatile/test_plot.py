import unittest
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
        hist3 = [(0.1,1,20), np.sin(x)] 
    ),
    axes = dict(
        xaxis = {"label": "xlabel", "scale": "log"},
        yaxis = {"label": "ylabel", "scale": "linear"}
    )
)

class TestViewer(unittest.TestCase):
    def test_plotly(self):
        vai.plot.Viewer(backend="plotly")
        vai.plot.Viewer(data=DATA, backend="plotly")
        vai.plot.Viewer(data=DATA, backend="plotly", template="seaborn")
        
    def test_mpl(self):
        vai.plot.Viewer(backend="mpl")
        vai.plot.Viewer(data=DATA, backend="mpl")
        vai.plot.Viewer(data=DATA, backend="mpl", template="seaborn")

    def test_uniplot(self):
        vai.plot.Viewer(backend="uniplot")
        vai.plot.Viewer(data=DATA, backend="uniplot", show_controls=False)
        vai.plot.Viewer(data=DATA, backend="uniplot", template="seaborn", show_controls=False)

    def test_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            vai.plot.Viewer(backend="other_backend")

    def test_add_update_plotly(self):
        v = vai.plot.Viewer(backend="plotly")
        v.add_line(x=DATA["line"]["line1"][0], y=DATA["line"]["line1"][1], name="line")
        v.add_scatter(x=DATA["scatter"]["scatter1"][0], y=DATA["scatter"]["scatter1"][1], name="scatter")
        v.add_histogram(data=DATA["scatter"]["scatter1"][1], bins=10, name="histogram1")
        v.add_histogram(data=DATA["scatter"]["scatter2"][1], bins=(0,1,10), name="histogram2")

        v.update_line(name="line", x=DATA["line"]["line2"][0], y=DATA["line"]["line2"][1])
        v.update_scatter(name="scatter", x=DATA["scatter"]["scatter2"][0], y=DATA["scatter"]["scatter2"][1])
        v.update_histogram(name="histogram1", data=DATA["scatter"]["scatter2"][1], bins=10)
        v.update_histogram(name="histogram2", data=DATA["scatter"]["scatter1"][1], bins=(0,1,10))

    def test_add_update_mpl(self):
        v = vai.plot.Viewer(backend="mpl")
        v.add_line(x=DATA["line"]["line1"][0], y=DATA["line"]["line1"][1], name="line")
        v.add_scatter(x=DATA["scatter"]["scatter1"][0], y=DATA["scatter"]["scatter1"][1], name="scatter")
        v.add_histogram(data=DATA["scatter"]["scatter1"][1], bins=10, name="histogram1")
        v.add_histogram(data=DATA["scatter"]["scatter2"][1], bins=(0,1,10), name="histogram2")

        v.update_line(name="line", x=DATA["line"]["line2"][0], y=DATA["line"]["line2"][1])
        v.update_scatter(name="scatter", x=DATA["scatter"]["scatter2"][0], y=DATA["scatter"]["scatter2"][1])
        v.update_histogram(name="histogram1", data=DATA["scatter"]["scatter2"][1], bins=10)
        v.update_histogram(name="histogram2", data=DATA["scatter"]["scatter1"][1], bins=(0,1,10))

    def test_add_update_uniplot(self):
        v = vai.plot.Viewer(backend="uniplot", show_controls=False)
        v.add_line(x=DATA["line"]["line1"][0], y=DATA["line"]["line1"][1], name="line")
        v.add_scatter(x=DATA["scatter"]["scatter1"][0], y=DATA["scatter"]["scatter1"][1], name="scatter")
        v.add_histogram(data=DATA["scatter"]["scatter1"][1], bins=10, name="histogram1")
        v.add_histogram(data=DATA["scatter"]["scatter2"][1], bins=(0,1,10), name="histogram2")

        v.update_line(name="line", x=DATA["line"]["line2"][0], y=DATA["line"]["line2"][1])
        v.update_scatter(name="scatter", x=DATA["scatter"]["scatter2"][0], y=DATA["scatter"]["scatter2"][1])
        v.update_histogram(name="histogram1", data=DATA["scatter"]["scatter2"][1], bins=10)
        v.update_histogram(name="histogram2", data=DATA["scatter"]["scatter1"][1], bins=(0,1,10))

    def test_getter_setter_plotly(self):
        v = vai.plot.Viewer(backend="plotly")
        v.get_figure()

        v.set_xlabel("x")
        v.set_ylabel("y")

        v.set_xscale("linear")
        v.set_xscale("log")
        v.set_yscale("linear")
        v.set_yscale("log")

    def test_getter_setter_mpl(self):
        v = vai.plot.Viewer(backend="mpl")
        v.get_figure()

        v.set_xlabel("x")
        v.set_ylabel("y")

        v.set_xscale("linear")
        v.set_xscale("log")
        v.set_yscale("linear")
        v.set_yscale("log")

    def test_getter_setter_uniplot(self):
        v = vai.plot.Viewer(backend="uniplot", show_controls=False)
        v.get_figure()

        v.set_xlabel("x")
        v.set_ylabel("y")

        v.set_xscale("linear")
        v.set_xscale("log")
        v.set_yscale("linear")
        v.set_yscale("log")

    def test_legend_plotly(self):
        v = vai.plot.Viewer(backend="plotly")
        v.show_legend(True)
        v.show_legend(False)

    def test_legend_mpl(self):
        v = vai.plot.Viewer(backend="mpl")
        v.show_legend(True)

        with self.assertRaises(NotImplementedError):
            v.show_legend(False)

class TestLine(unittest.TestCase):
    def test_plotly(self):
        vai.plot.Line(y=DATA["line"]["line1"][1], 
                      backend="plotly")
        vai.plot.Line(x=DATA["line"]["line1"][0], 
                      y=DATA["line"]["line1"][1], 
                      backend="plotly")
        vai.plot.Line(y={name: data[1] for name, data in DATA["line"].items()},
                      backend="plotly")
        vai.plot.Line(y={name: data[1] for name, data in DATA["line"].items()},
                      xscale="log", yscale="log", xlabel="xlabel", ylabel="ylabel",
                      backend="plotly")
        vai.plot.Line(y=DATA["line"]["line1"][1], template="seaborn", backend="plotly")
        
    def test_mpl(self):
        vai.plot.Line(y=DATA["line"]["line1"][1], backend="mpl")
        vai.plot.Line(x=DATA["line"]["line1"][0], 
                      y=DATA["line"]["line1"][1], 
                      backend="mpl")
        vai.plot.Line(y={name: data[1] for name, data in DATA["line"].items()}, 
                      backend="mpl")
        vai.plot.Line(y={name: data[1] for name, data in DATA["line"].items()},
                      xscale="log", yscale="log", xlabel="xlabel", ylabel="ylabel", 
                      backend="mpl")
        vai.plot.Line(y=DATA["line"]["line1"][1], template="seaborn", backend="mpl")

    def test_uniplot(self):
        vai.plot.Line(y=DATA["line"]["line1"][1], backend="uniplot", show_controls=False)
        vai.plot.Line(x=DATA["line"]["line1"][0], 
                      y=DATA["line"]["line1"][1], 
                      backend="uniplot", show_controls=False)
        vai.plot.Line(y={name: data[1] for name, data in DATA["line"].items()}, 
                      backend="uniplot", show_controls=False)
        vai.plot.Line(y={name: data[1] for name, data in DATA["line"].items()},
                      xlabel="xlabel", ylabel="ylabel", 
                      backend="uniplot", show_controls=False)
        vai.plot.Line(y=DATA["line"]["line1"][1], template="seaborn", backend="uniplot", show_controls=False)

class TestScatter(unittest.TestCase):
    def test_plotly(self):
        vai.plot.Scatter(y=DATA["scatter"]["scatter1"][1], backend="plotly")
        vai.plot.Scatter(x=DATA["scatter"]["scatter1"][0], 
                      y=DATA["scatter"]["scatter1"][1], backend="plotly")
        vai.plot.Scatter(y={name: data[1] for name, data in DATA["scatter"].items()}, 
                         backend="plotly")
        vai.plot.Scatter(y={name: data[1] for name, data in DATA["scatter"].items()},
                      xscale="log", yscale="log", xlabel="xlabel", ylabel="ylabel", backend="plotly")
        vai.plot.Scatter(y=DATA["scatter"]["scatter1"][1], template="seaborn",
                         backend="plotly")

    def test_mpl(self):
        vai.plot.Scatter(y=DATA["scatter"]["scatter1"][1], backend="mpl")
        vai.plot.Scatter(x=DATA["scatter"]["scatter1"][0], 
                      y=DATA["scatter"]["scatter1"][1],
                      backend="mpl")
        vai.plot.Scatter(y={name: data[1] for name, data in DATA["scatter"].items()},
                         backend="mpl")
        vai.plot.Scatter(y={name: data[1] for name, data in DATA["scatter"].items()},
                      xscale="log", yscale="log", xlabel="xlabel", ylabel="ylabel",
                      backend="mpl")
        vai.plot.Scatter(y=DATA["scatter"]["scatter1"][1], template="seaborn", 
                         backend="mpl")
        
    def test_uniplot(self):
        vai.plot.Scatter(y=DATA["scatter"]["scatter1"][1], backend="uniplot", show_controls=False)
        vai.plot.Scatter(x=DATA["scatter"]["scatter1"][0], 
                      y=DATA["scatter"]["scatter1"][1],
                      backend="uniplot", show_controls=False)
        vai.plot.Scatter(y={name: data[1] for name, data in DATA["scatter"].items()},
                         backend="uniplot", show_controls=False)
        vai.plot.Scatter(y={name: data[1] for name, data in DATA["scatter"].items()},
                      xlabel="xlabel", ylabel="ylabel",
                      backend="uniplot", show_controls=False)
        vai.plot.Scatter(y=DATA["scatter"]["scatter1"][1], template="seaborn", 
                         backend="uniplot", show_controls=False)

class TestHistogram(unittest.TestCase):
    def test_plotly(self):
        vai.plot.Histogram(data=DATA["scatter"]["scatter1"][1], backend="plotly")
        vai.plot.Histogram(data=DATA["scatter"]["scatter1"][1], bins=10, 
                           backend="plotly")
        vai.plot.Histogram(data=DATA["scatter"]["scatter1"][1], bins=(0,1,10),
                           backend="plotly")
        vai.plot.Histogram(data=dict(first=DATA["scatter"]["scatter1"][1],
                                     second=DATA["scatter"]["scatter2"][1]), 
                           bins=(0,1,10), backend="plotly")
        vai.plot.Histogram(data=DATA["scatter"]["scatter1"][1], bins=10, template="seaborn", backend="plotly")
        vai.plot.Histogram(data=DATA["scatter"]["scatter1"][1], 
                           xscale="log", yscale="log", xlabel="xlabel", ylabel="ylabel", 
                           backend="plotly")
        
        with self.assertRaises(TypeError):
            vai.plot.Histogram(data=DATA["scatter"]["scatter1"][1], bins="nonsense", 
                               backend="plotly")
        with self.assertRaises(TypeError):
            vai.plot.Histogram(data=DATA["scatter"]["scatter1"][1], bins=(1,2), 
                               backend="plotly")

    def test_mpl(self):
        vai.plot.Histogram(data=DATA["scatter"]["scatter1"][1], backend="mpl")
        vai.plot.Histogram(data=DATA["scatter"]["scatter1"][1], bins=10, backend="mpl")
        vai.plot.Histogram(data=DATA["scatter"]["scatter1"][1], bins=(0,1,10), 
                           backend="mpl")
        vai.plot.Histogram(data=dict(first=DATA["scatter"]["scatter1"][1],
                                     second=DATA["scatter"]["scatter2"][1]), 
                           bins=(0,1,10), backend="mpl")
        vai.plot.Histogram(data=DATA["scatter"]["scatter1"][1], bins=10, template="seaborn",
                           backend="mpl")
        vai.plot.Histogram(data=DATA["scatter"]["scatter1"][1], 
                           xscale="log", yscale="log", xlabel="xlabel", ylabel="ylabel",
                           backend="mpl")
        
        with self.assertRaises(TypeError):
            vai.plot.Histogram(data=DATA["scatter"]["scatter1"][1], bins="nonsense", 
                               backend="mpl") 
        with self.assertRaises(TypeError):
            vai.plot.Histogram(data=DATA["scatter"]["scatter1"][1], bins=(1,2), 
                               backend="mpl")
            
    def test_uniplot(self):
        vai.plot.Histogram(data=DATA["scatter"]["scatter1"][1], backend="mpl")
        vai.plot.Histogram(data=DATA["scatter"]["scatter1"][1], bins=10, backend="mpl")
        vai.plot.Histogram(data=DATA["scatter"]["scatter1"][1], bins=(0,1,10), 
                           backend="uniplot", show_controls=False)
        vai.plot.Histogram(data=dict(first=DATA["scatter"]["scatter1"][1],
                                     second=DATA["scatter"]["scatter2"][1]), 
                           bins=(0,1,10), backend="uniplot", show_controls=False)
        vai.plot.Histogram(data=DATA["scatter"]["scatter1"][1], bins=10, template="seaborn",
                           backend="uniplot", show_controls=False)
        vai.plot.Histogram(data=DATA["scatter"]["scatter1"][1], 
                           xscale="log", xlabel="xlabel", ylabel="ylabel",
                           backend="uniplot", show_controls=False)
        
        with self.assertRaises(TypeError):
            vai.plot.Histogram(data=DATA["scatter"]["scatter1"][1], bins="nonsense", 
                               backend="uniplot", show_controls=False) 
        with self.assertRaises(TypeError):
            vai.plot.Histogram(data=DATA["scatter"]["scatter1"][1], bins=(1,2), 
                               backend="uniplot", show_controls=False)

class TestPreview(unittest.TestCase):
    def test_plotly(self):
        ...
        
    def test_mpl(self):
        ...

    def test_uniplot(self):
        ...

if __name__ == '__main__':
    unittest.main()