try:
    import uniplot
except ImportError:
    uniplot = None

## https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook ###
def auto_backend():
    # If uniplot is not installed, use plotly (will result in a dictionary output)
    if uniplot is None: return "plotly"
    
    try:
        from IPython import get_ipython
        if 'IPKernelApp' not in get_ipython().config: return "uniplot"
    except ImportError: return "uniplot"
    except AttributeError: return "uniplot"
    return "plotly"

class EmptyRep:
    # Helper Class that generates empty cell output
    def __repr__(self):
        return ""

#####################
# Plot Base Classes #
#####################