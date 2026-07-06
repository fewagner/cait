import os

import pytest

from ._helper import cd, execute_notebook, get_absolute_path
from .fixtures import tempdir_fnc

# THIS FILE RUNS ALL CELLS OF THE TUTORIAL NOTEBOOKS TO TEST FOR ERRORS

@pytest.mark.parametrize("notebook", [
    "00tutorial_trigger",
    "01datahandler_baseFeatures",
    "02sev_nps_of_blres",
    "03amplitude_reconstruction",
    "04energy_calibration",
    "05efficiency_simulation",
    # Add all tutorial notebooks that should run here:
    # Without file extension and relative to docs/source/tutorials/
    # NOTE: this runs the notebooks separately. I.e. variables, values, data
    # is not preserved between notebooks.
    ]
)
def test_tutorial(tempdir_fnc, notebook):
    notebook_path = get_absolute_path(
        "..", 
        "docs", 
        "source", 
        "tutorials", 
        notebook+".ipynb",
        )

    # Make a separate folder for every notebook (same name as the notebook)
    # to prevent interaction between files created by the notebooks.
    temp_dir = os.path.join(tempdir_fnc.name, notebook)
    os.mkdir(temp_dir)

    # Change to working directory of notebook and execute it.
    with cd(temp_dir):
        execute_notebook(notebook_path)