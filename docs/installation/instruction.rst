######################################
Installation instruction
######################################

*********************************************************
Installation
*********************************************************

Start with a new Python environment:

.. code-block:: bash

    conda create -n pykitPIV python=3.10

.. code-block:: bash

    conda activate pykitPIV

Install requirements:

.. code-block:: bash

    pip install numpy
    pip install jupyterlab
    pip install matplotlib
    pip install h5py
    pip install torch
    pip install pandas
    pip install scipy
    pip install cmcrameri

Additional requirements for the ``ml`` module:

.. code-block:: bash

    pip install torchvision
    pip install tensorflow
    pip install tf-agents
    pip install gymnasium
    pip install tqdm
    pip install onnxruntime
    pip install pygame

Optional for local documentation builds:

.. code-block:: bash

    pip install Sphinx
    pip install sphinxcontrib-bibtex
    pip install furo

Clone the pykitPIV repository:

.. code-block:: bash

    git clone https://gitlab.empa.ch/kamila.zdybal/pykitPIV.git

and move there:

.. code-block:: bash

    cd pykitPIV

Install ``pykitPIV``:

.. code-block:: bash

    python -m pip install .

*********************************************************
Local documentation build
*********************************************************

Build documentation:

.. code-block:: bash

    cd docs
    sphinx-build -b html . builddir
    make html

Open documentation in a web browser:

.. code-block:: bash

    open _build/html/index.html

*********************************************************
Unit tests
*********************************************************

To run unit tests, run the following in the main ``pykitPIV`` directory:

.. code-block:: bash

    python -m unittest discover -v
