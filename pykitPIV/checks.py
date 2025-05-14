import numpy as np
import warnings

def check_positive_int_or_float(x, name):

    if not isinstance(x, int) and not isinstance(x, float):
        raise ValueError("Parameter `" + name + "` has to be of type 'int' or 'float'.")

    if x <= 0:
        raise ValueError("Parameter `" + name + "` has to be a non-zero, positive value.")

def check_non_negative_int_or_float(x, name):

    if not isinstance(x, int) and not isinstance(x, float):
        raise ValueError("Parameter `" + name + "` has to be of type 'int' or 'float'.")

    if x < 0:
        raise ValueError("Parameter `" + name + "` has to be a non-negative value.")

def check_two_element_tuple(x, name):

    if not isinstance(x, tuple):
        raise ValueError("Parameter `" + name + "` has to be of type 'tuple'.")

    if len(x) != 2:
        raise ValueError("Parameter `" + name + "` has to have two elements.")

def check_min_max_tuple(x, name):

    if not x[0] <= x[1]:
        raise ValueError("The first element in `" + name + "` has to be smaller than or equal to the second element.")

def check_four_dimensional_2D_vector_field_tensor(vector_field, name):

    # This is a check for a `numpy.ndarray` that has a generic shape (N, 2, H, W).

    if not isinstance(vector_field, np.ndarray):
        raise ValueError("Parameter `" + name + "` has to be of type 'numpy.ndarray'.")

    if vector_field.ndim != 4:
        raise ValueError("Parameter `" + name + "` has to be a four-dimensional tensor.")

    (N, Ch, H, W) = np.shape(vector_field)

    if Ch != 2:
        raise ValueError("Parameter `" + name + "` has to contain two vector field components.")

def check_four_dimensional_1D_scalar_tensor(vector_field, name):

    # This is a check for a `numpy.ndarray` that has a generic shape (N, 1, H, W).

    if not isinstance(vector_field, np.ndarray):
        raise ValueError("Parameter `" + name + "` has to be of type 'numpy.ndarray'.")

    if vector_field.ndim != 4:
        raise ValueError("Parameter `" + name + "` has to be a four-dimensional tensor.")

    (N, Ch, H, W) = np.shape(vector_field)

    if Ch != 1:
        raise ValueError("Parameter `" + name + "` has to contain two vector field components.")
