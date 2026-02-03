from __future__ import annotations
from typing import Sequence, Optional
from scipy.optimize import least_squares
import numpy as np

from base import Model, HasTransfrom, HasJacobian, HasInitialGuess

def least_squares_optimizer(model:Model,*,
                            x0:Optional[Sequence[float]]=None,
                            lower_bounds:Optional[Sequence[float]]=None,
                            upper_bounds:Optional[Sequence[float]]=None,
                            xtol = 1e-12,
                            ftol = 1e-12,
                            gtol = 1e-12,
                            max_nfev = 3000,
                            ):
    """
    本函数输出结果基于scipy.optimize.least_squares的输出，
    注意，res.x用于储存优化时迭代计算的中间参数，有物理意义的参数实际储存在res.set中，是一个集合。
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


    res = least_squares(model.residual, x0, jac=jac, bounds=bounds, 
                         method='trf', xtol=xtol, ftol=ftol, gtol=gtol, max_nfev=max_nfev,
                    )

    if isinstance(model, HasTransfrom):
        sol = model.to_physics(res.x)
    else:
        sol = res.x
    if isinstance(sol, set):
        res.set = sol
    else:
        res.set = set()
        res.set.add(res.x)
    
    return res