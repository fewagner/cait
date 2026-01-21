import os

import pytest

from ._helper import cd, execute_notebook, get_absolute_path
from .fixtures import tempdir

# THIS FILE RUNS ALL CELLS OF THE TUTORIAL NOTEBOOKS TO TEST FOR ERRORS

@pytest.mark.parametrize("notebook", [
    "tutorial_trigger",
    # Add all tutorial notebooks that should run here:
    # Without file extension and relative to docs/source/tutorials/
    # NOTE: this runs the notebooks separately. I.e. variables, values, data
    # is not preserved between notebooks.
    ]
)
def test_tutorial(tempdir, notebook):
    notebook_path = get_absolute_path(
        "..", 
        "docs", 
        "source", 
        "tutorials", 
        notebook+".ipynb",
        )

    # Make a separate folder for every notebook (same name as the notebook)
    # to prevent interaction between files created by the notebooks.
    temp_dir = os.path.join(tempdir.name, notebook)
    os.mkdir(temp_dir)

    # Change to working directory of notebook and execute it.
    with cd(temp_dir):
        execute_notebook(notebook_path)