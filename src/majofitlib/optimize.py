from __future__ import annotations
from typing import Sequence, Optional
import scipy
import numpy as np

from .base import Model, HasTransfrom, HasJacobian, HasInitialGuess

def least_squares(model:Model,*,
                            x0:Optional[Sequence[float]]=None,
                            lower_bounds:Optional[Sequence[float]]=None,
                            upper_bounds:Optional[Sequence[float]]=None,
                            xtol = 1e-12,
                            ftol = 1e-12,
                            gtol = 1e-12,
                            max_nfev = 3000,
                            ):
    """
    The return value (res) of this function is based on the return value of scipy.optimize.least_squares.
    Note that res.x is used to store the intermediate parameters for iterative calculations during optimization. 
    Parameters with physical significance are actually stored in res.set, which is a collection.

    The parameter x0 takes true physical quantities;
    But lower_bounds and upper_bounds take the bound of intermediate parameters during calculations.
    Do not specify the bounds unless you understand the meaning of the intermediate parameters.
    """
    if not x0:
        if isinstance(model, HasInitialGuess):
            x0 = model.initial_guess()
        else:
            raise TypeError(f"Model {model.__class__.__name__} needs to choose initial parameters manually.")

    if isinstance(model, HasJacobian):
        jac = model.jacobian
    else:
        jac = 'cs'
 
    if not lower_bounds:
        lower_bounds = (-np.inf for _ in range(len(x0)))
    if not upper_bounds:
        upper_bounds = (np.inf for _ in range(len(x0)))
    
    bounds = (tuple(lower_bounds), tuple(upper_bounds))


    res = scipy.optimize.least_squares(model.residual, x0, jac=jac, bounds=bounds, 
                         method='trf', xtol=xtol, ftol=ftol, gtol=gtol, max_nfev=max_nfev,
                    )

    if isinstance(model, HasTransfrom):
        res.x_set = model.to_physics(res.x)
    else:
        res.x_set = set().add(res.x)
    
    res.forwards = [model.forward_factory(x) for x in res.x_set]
    res.datas = [forward(model.data_x) for forward in res.forwards]

    return res