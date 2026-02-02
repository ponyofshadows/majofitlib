from __future__ import annotations
from typing import Iterable, Optional
from scipy.optimize import least_squares
import numpy as np

from base import *

def least_squares_optimizer(model:ModelStrategy,*,
                            x0:Optional[Iterable[float]]=None,
                            lower_bound:Optional[Iterable[float]]=None,
                            upper_bound:Optional[Iterable[float]]=None):
    """
    本函数输出结果基于scipy.optimize.least_squares的输出，
    注意，res.x用于储存优化时迭代计算的中间参数，有物理意义的参数实际储存在res.set中，是一个集合。
    注意，如果指定初值或bound（取值范围），为了保证普适性，只能指定拟合函数实际使用的参数的取值范围。
    """
    method = "trf"
    xtol = 1e-12
    ftol = 1e-14
    gtol = 1e-12
    max_nfev = 5000

    if not x0:
        if isinstance(model, HasInitialGuess):
            x0 = model.initial_guess()
        else:
            raise TypeError(f"Model {model.__class__.__name__} needs to choose initial parameters manually.")

    if isinstance(model, HasJacobian):
        jac = model.jacobian
    else:
        jac = 'cs'
 
    if not lower_bound:
        lower_bound = (-np.inf for _ in range(len(x0)))
    if not upper_bound:
        upper_bound = (np.inf for _ in range(len(x0)))
    
    bounds = (tuple(lower_bound), tuple(upper_bound))


    res = least_squares(model.residual, x0, jac=jac, bounds=bounds, 
                         method=method, xtol=xtol, ftol=ftol, gtol=gtol, max_nfev=max_nfev,
                        x_scale="jac",
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