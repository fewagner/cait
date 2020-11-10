import matplotlib as mpl

def set_mpl_backend_pgf():
    mpl.use("pgf")
    mpl.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'sans-serif',
        'font.size' : 12,
        'text.usetex': True,
        'pgf.rcfonts': False,
    })

def set_mpl_backend_fontsize(fontsize):
    mpl.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'sans-serif',
        'font.size' : fontsize,
        'text.usetex': True,
        'pgf.rcfonts': False,
    })
