from __future__ import annotations
from typing import Protocol, runtime_checkable, Callable, Optional, NamedTuple
from numpy.typing import NDArray
from collections.abc import MutableMapping
import numpy as np


registered_units = {}
registered_models = []

class Unit(NamedTuple):
    name: str
    multiplier: float
    unicode_name: str

    def __str__(self):
        return self.unicode_name
    def __truediv__(self, other)->float:
        if isinstance(other, Unit):
            return self.multiplier / other.multiplier
        else:
            return NotImplemented

class PhysicalArray(NamedTuple):
    array: NDArray
    unit: Unit

class Data(MutableMapping):
    __slot__ = ("_items",)
    def __init__(self, init:Optional[dict|Data]=None):
        if init:
            if isinstance(init, Data):
                self._items = dict(init._items)
            elif isinstance(init, dict):
                for key, value in init.items():
                    self[key] = value
            else:
                raise TypeError(f"class Data needs a dict to initialize, not {type(init)}.")
        else:
            self._items = {}
    
    def __getitem__(self, key):
        return self._items[key]
    
    def __setitem__(self, key, value):
        if isinstance(key, str):
            try:
                physical_quantity, unit_name = [ s.strip() for s in key.split(",")[0:2] ]
            except ValueError:
                raise ValueError("The key of Data should be a string of comma-separated physical quantities and units." \
                    "For example: 'H, T'")
        else:
            raise TypeError(f"The key of Data should be a string of comma-separated physical quantities and units', not {type(key)}")
        
        try:
            for unit in registered_units[physical_quantity]:
                if unit_name == unit.name:
                    matched_unit = unit
                    break
            else:
                raise ValueError(f"Unit '{unit_name}' is not found. Here are available units of {physical_quantity}:\n"+
                                 '\n'.join(f"\t -{unit.name}" for unit in registered_units))
        except KeyError:
            raise ValueError(f"Physical quantity '{physical_quantity}' is not found.")

        self._items[physical_quantity] = PhysicalArray(np.asarray(value), matched_unit)
    
    def __delitem__(self, key):
        del self._items[key]
    
    def __iter__(self):
        return iter(self._items)

    def __len__(self):
        return len(self._items)
    
    def __repr__(self):
        return f"Data({'\n'.join([f'{key}={value}' for key,value in self._items.items()])})"
    


class Forward(Protocol):
    def __call__(self, x):
        ...

class Model(Protocol):
    headers: set[str]
    forward: Forward
    def residual(self, x):
        ...
@runtime_checkable
class HasJacobian(Protocol):
    def jacobian(self, x):
        ...
@runtime_checkable
class HasInitialGuess(Protocol):
    def initial_guess(self):
        ...
@runtime_checkable
class HasTransfrom(Protocol):
    def from_physics(self, x):
        ...
    def to_physics_set(self, x:tuple)->set[tuple]:
        ...



def unit_register(physical_quantity:str, unit_name:str, multiplier:float=1, unicode_name:Optional[str]=None):
    if unicode_name is None:
        unicode_name = unit_name
    registered_units.setdefault(physical_quantity, set()).add(
        Unit(
            name = unit_name,
            multiplier= multiplier,
            unicode_name=unicode_name,
        )
    )


def model_register(model:Model)->Model:
    registered_models.append(model)
    return model