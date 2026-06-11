# %% Libs:
import numpy as np

# %% Round2mult function:
def Round2mult(num, mult, method='up'):
    '''
    Round a number to the nearest multiple of a specified value, using ceiling (up) or flooring (down).

    Summary:
    This function takes a number and rounds it to the nearest multiple of a specified value, 
    either rounding up (ceiling) or down (flooring) based on the chosen method.

    Parameters:
    - num (float): The number to be rounded.
    - mult (int or float): The multiple to which 'num' should be rounded.
    - method (str): The rounding method to use ('up' for rounding up, 'down' for rounding down).
                    Default is 'ceil'.

    Returns:
    - float: The nearest multiple of 'mult' based on the specified rounding method.

    Example:
    num_rounded_up = Round2mult(num, mult, method='up')
    num_rounded_down = Round2mult(num, mult, method='down')
    
    Created/Last modified: 2024-11-01
    '''
    if method == 'up':
        return np.ceil(num / mult) * mult
    elif method == 'down':
        return np.floor(num / mult) * mult
    else:
        raise ValueError("Invalid method. Use 'up' or 'down'.")
