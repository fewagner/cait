# HELPER FUNCTIONS USED MAINLY TO AUTOMATICALLY EXECUTE TUTORIAL/MANUAL
# TEST NOTEBOOKS AND SCRIPTS
import json
import os
import warnings
from typing import List

import matplotlib as mpl
import matplotlib.pyplot as plt

import cait as ai


def get_absolute_path(*path_relative_to_ai):
    return os.path.join( os.path.dirname(ai.__file__), *path_relative_to_ai )
 
def exec_blocks(lines: List[str], filepath: str):
    # Code blocks with indentation (e.g. for-loops) have to be 
    # passed to exec() as a single string. Therefore, we collect
    # such blocks here before calling exec().

    # We also have to make sure that the local variables are 
    # maintained between calls to exec() (this worked 'by chance'
    # in Python versions<3.13).
    locals_dict = {}
    # We have to track how many (, [, and { are still open, because
    # they could span several lines.
    if not lines[0].strip().startswith("#"):
        b_round = lines[0].count("(") - lines[0].count(")")
        b_square = lines[0].count("[") - lines[0].count("]")
        b_curly =lines[0].count("{") - lines[0].count("}")
    else: 
        b_round, b_square, b_curly = 0, 0, 0

    block = lines[0] + "\n"

    for i in range(1, len(lines)):
        if i < len(lines) and ( 
            # Check if line is indented ...
            lines[i].startswith("    ") 
            # ... or any brackets are still open ...
            or ( b_round + b_square + b_curly > 0 ) 
            ):
            # ... and append to the block. This has to be done
            # because exec() requires complete statements
            block += lines[i] + "\n"
        else:
            # Execute block (and format occurring errors nicely).
            try:
                # Suppress figures from popping up.
                with SuppressPlots():
                    exec(block, {**globals(), **locals_dict}, locals_dict)
                    plt.close("all")

            except Exception as e:
                print(f"\n\033[1mFrom {filepath}:\033[0m")
                print(f"\033[91mIn line {i}:\033[0m")
                print("".join(lines[max(0, i-5):i-1]), end="\r")
                print(f"\033[91m>{lines[i-1]}\033[0m")
                print(f"\n\033[91m{type(e).__name__}: {e}\033[0m")
                raise e
            
            if i < len(lines):
                block = lines[i]

        # If the line is not a comment, count the net change in number of brackets.
        if not lines[i].strip().startswith("#"):
            b_round += lines[i].count("(") - lines[i].count(")")
            b_square += lines[i].count("[") - lines[i].count("]")
            b_curly += lines[i].count("{") - lines[i].count("}")

# Read every code cell of a jupyter notebook and execute it.
# As suggested by https://stackoverflow.com/a/68728574
def execute_notebook(notebook_path: str):
    with open(notebook_path) as f:
        nb = json.load(f)

    lines = []
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            lines += [line for line in cell["source"] if not line.startswith("%")] + ["\n"]

    exec_blocks(lines, notebook_path)

# Execute a script by reading its lines.
def execute_script(script_path: str):
    with open(script_path) as f:
        lines = f.readlines()

    exec_blocks(lines, script_path)

# Convoluted but good-practice-way of changing the directory.
# (needed to find files that are specified with relative paths
# in the notebooks).
# As suggested by https://stackoverflow.com/a/13197763
class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)

class SuppressPlots:
    """Context manager that suppresses matplotlib plots during testing."""
    def __init__(self):
        self.old_backend = mpl.get_backend()

    def __enter__(self):
        mpl.use("Agg")
        warnings.filterwarnings("ignore", "Matplotlib is currently using agg")
        warnings.filterwarnings("ignore", "FigureCanvasAgg is non-interactive, and thus cannot be shown")
        # Also force to NOT use LaTeX (might not be installed on testing host).
        mpl.rcParams["text.usetex"] = False

    def __exit__(self, etype, value, traceback):
        mpl.use(self.old_backend)