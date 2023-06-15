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
        with self.assertRaises(NotImplementedError):
            vai.plot.Viewer(backend="mpl")

    def test_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            vai.plot.Viewer(backend="other_backend")

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
        with self.assertRaises(NotImplementedError):
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
        with self.assertRaises(NotImplementedError):
            vai.plot.Scatter(y=DATA["scatter"]["scatter1"][1], backend="mpl")

class TestPreview(unittest.TestCase):
    def test_plotly(self):
        ...
        
    def test_mpl(self):
        ...

if __name__ == '__main__':
    unittest.main()