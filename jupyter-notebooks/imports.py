import numpy as np
import pandas as pd
import time
import random
import csv
import copy as cp
from tqdm import tqdm
import heapq
import h5py
import pickle
import cmcrameri.cm as cmc
from scipy.stats import norm
import os
from platform import python_version

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
from matplotlib import cm
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap

from mpl_toolkits import mplot3d
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from  matplotlib import rc

cmap = cmc.lajolla

print('Python==' + python_version())
print()
print('numpy==' + np.__version__)
print('pandas==' + pd.__version__)

import pypiv
