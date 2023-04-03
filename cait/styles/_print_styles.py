def txt_fmt(text: str, color: str, style: str = None):
    """
    Format output string with color and style.

    :param text: The text to format
    :type text: str
    :param color: The color to use. Valid options are "purple", "cyan", "darkcyan", "blue", "green", "yellow" and "red".
    :type color: str
    :param style: Optional. Valid options are "underline" and "bold".
    :type style: str
    """
    colors = {"purple": '\033[95m', 
              "cyan": '\033[96m', 
              "darkcyan": '\033[36m', 
              "blue": '\033[94m', 
              "green": '\033[92m',
              "yellow": '\033[93m', 
              "red": '\033[91m'}
    styles = {"underline": '\033[4m', 
              "bold": '\033[1m'}
    end = '\033[0m'
    
    return styles[style]+colors[color]+text+end if style is not None else colors[color]+text+end

def fmt_gr(text):
    """
    Format HDF5 group names consistently.

    :param text: The group name to format
    :type text: str
    """
    return txt_fmt(text, "purple", "bold")

def fmt_ds(text):
    """
    Format HDF5 dataset names consistently.

    :param text: The dataset name to format
    :type text: str
    """
    return txt_fmt(text, "darkcyan", "bold")