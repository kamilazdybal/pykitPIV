import numpy as np
import warnings

def check_two_element_tuple(x, name):

    if not isinstance(x, tuple):
        raise ValueError("Parameter `" + name + "` has to be of type 'tuple'.")

    if len(x) != 2:
        raise ValueError("Parameter `" + name + "` has to have two elements.")

def check_min_max_tuple(x, name):

    if not x[0] <= x[1]:
        raise ValueError("The first element in `" + name + "` has to be smaller than or equal to the second element.")