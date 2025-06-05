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
    pip install matplotlib
    pip install h5py
    pip install pandas
    pip install scipy==1.15.1
    pip install cmcrameri
    pip install urllib3
    pip install termcolor
    pip install scikit-learn
    pip install torch==2.2.2
    pip install torchvision==0.17.2
    pip install tensorflow==2.16.2
    pip install keras==3.8.0
    pip install gymnasium
    pip install tqdm

Additional requirements for Jupyter notebook tutorials:

.. code-block:: bash

    pip install --upgrade jupyterlab
    pip install onnxruntime

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
