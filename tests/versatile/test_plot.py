import unittest
import numpy as np
import cait.versatile as vai

x = np.linspace(-1,1,100)

DATA = dict(
    line = dict(
        line1 = [ x , np.sin(x) ],
        line2 = [ x , np.cos(x) ]
    ),
    scatter = dict(
        scatter1 = [ x , x**2 ],
        scatter2 = [ x , -x ]
    ),
    axes = dict(
        xaxis = {"label": "xlabel", "scale": "linear"},
        yaxis = {"label": "ylabel", "scale": "log"}
    )
)

class TestViewer(unittest.TestCase):
    def test_plotly(self):
        vai.plot.Viewer(backend="plotly")
        vai.plot.Viewer(data=DATA, backend="plotly")
        vai.plot.Viewer(data=DATA, backend="plotly", template="seaborn")
        
    def test_mpl(self): # has to be implemented
        with self.assertRaises(TypeError):
            vai.plot.Viewer(backend="mpl")

    def test_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            vai.plot.Viewer(backend="other_backend")

    def test_add_update(self):
        v = vai.plot.Viewer(backend="plotly")
        v.add_line(x=DATA["line"]["line1"][0], y=DATA["line"]["line1"][1], name="line")
        v.add_scatter(x=DATA["scatter"]["scatter1"][0], y=DATA["scatter"]["scatter1"][1], name="scatter")
        v.add_histogram(data=DATA["scatter"]["scatter1"][1], bins=10, name="histogram1")
        v.add_histogram(data=DATA["scatter"]["scatter2"][1], bins=(0,1,10), name="histogram2")

        v.update_line(name="line", x=DATA["line"]["line2"][0], y=DATA["line"]["line2"][1])
        v.update_scatter(name="scatter", x=DATA["scatter"]["scatter2"][0], y=DATA["scatter"]["scatter2"][1])
        v.update_histogram(name="histogram1", data=DATA["scatter"]["scatter2"][1], bins=10)
        v.update_histogram(name="histogram2", data=DATA["scatter"]["scatter1"][1], bins=(0,1,10))

    def test_getter_setter(self):
        v = vai.plot.Viewer(backend="plotly")
        v.get_figure()

        v.set_xlabel("x")
        v.set_ylabel("y")

        v.set_xscale("linear")
        v.set_xscale("log")
        v.set_yscale("linear")
        v.set_yscale("log")

    def test_legend(self):
        v = vai.plot.Viewer(backend="plotly")
        v.show_legend(True)
        v.show_legend(False)

class TestLine(unittest.TestCase):
    def test_plotly(self):
        vai.plot.Line(y=DATA["line"]["line1"][1])
        vai.plot.Line(x=DATA["line"]["line1"][0], 
                      y=DATA["line"]["line1"][1])
        vai.plot.Line(y={name: data[1] for name, data in DATA["line"].items()})
        vai.plot.Line(y={name: data[1] for name, data in DATA["line"].items()},
                      xscale="log", yscale="log", xlabel="xlabel", ylabel="ylabel")
        vai.plot.Line(y=DATA["line"]["line1"][1], template="seaborn")
        
    def test_mpl(self): # has to be implemented
        with self.assertRaises(TypeError):
            vai.plot.Line(y=DATA["line"]["line1"][1], backend="mpl")

class TestScatter(unittest.TestCase):
    def test_plotly(self):
        vai.plot.Scatter(y=DATA["scatter"]["scatter1"][1])
        vai.plot.Scatter(x=DATA["scatter"]["scatter1"][0], 
                      y=DATA["scatter"]["scatter1"][1])
        vai.plot.Scatter(y={name: data[1] for name, data in DATA["scatter"].items()})
        vai.plot.Scatter(y={name: data[1] for name, data in DATA["scatter"].items()},
                      xscale="log", yscale="log", xlabel="xlabel", ylabel="ylabel")
        vai.plot.Scatter(y=DATA["scatter"]["scatter1"][1], template="seaborn")

    def test_mpl(self): # has to be implemented
        with self.assertRaises(TypeError):
            vai.plot.Scatter(y=DATA["scatter"]["scatter1"][1], backend="mpl")

class TestHistogram(unittest.TestCase):
    def test_plotly(self):
        vai.plot.Histogram(data=DATA["scatter"]["scatter1"][1])
        vai.plot.Histogram(data=DATA["scatter"]["scatter1"][1], bins=10)
        vai.plot.Histogram(data=DATA["scatter"]["scatter1"][1], bins=(0,1,10))
        vai.plot.Histogram(data=dict(first=DATA["scatter"]["scatter1"][1],
                                     second=DATA["scatter"]["scatter2"][1]), 
                           bins=(0,1,10))
        vai.plot.Histogram(data=DATA["scatter"]["scatter1"][1], bins=10, template="seaborn")
        vai.plot.Histogram(data=DATA["scatter"]["scatter1"][1], 
                           xscale="log", yscale="log", xlabel="xlabel", ylabel="ylabel")
        
        with self.assertRaises(TypeError):
            vai.plot.Histogram(data=DATA["scatter"]["scatter1"][1], bins="nonsense")
        with self.assertRaises(TypeError):
            vai.plot.Histogram(data=DATA["scatter"]["scatter1"][1], bins=(1,2))

    def test_mpl(self): # has to be implemented
        with self.assertRaises(TypeError):
            vai.plot.Histogram(data=DATA["scatter"]["scatter1"][1], backend="mpl")

class TestPreview(unittest.TestCase):
    def test_plotly(self):
        ...
        
    def test_mpl(self):
        ...

if __name__ == '__main__':
    unittest.main()