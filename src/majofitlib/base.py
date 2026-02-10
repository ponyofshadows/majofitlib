from __future__ import annotations
from typing import Protocol, runtime_checkable, Callable, Optional, NamedTuple, Iterable
from numpy.typing import NDArray
from collections.abc import MutableMapping
from dataclasses import dataclass, field
import numpy as np


registered_units = {}
registered_models = {}

class Unit(NamedTuple):
    physical_quantity: str
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
    
    def convert(self, other_unit_name)->float:
        return registered_units[self.physical_quantity][other_unit_name] / self



@dataclass(slots=True)
class Column:
    name: str
    array: NDArray
    unit: Unit
    col_type: str
    _allowed_types:tuple = field(init=False, default=("X","Y","PARA"))
    
    def __post_init__(self):
        if self.col_type not in self._allowed_types:
            raise ValueError(f"Illegal col_type:{self.col_type}. Supported options:{self._allowed_types}")

class Data(MutableMapping):
    __slots__ = ("_items")
    def __init__(self, init:Optional[dict|Data]=None):
        if init:
            if isinstance(init, Data):
                self._items = dict(init._items)
            elif isinstance(init, dict):
                self._items = {}
                for key, value in init.items():
                    self[key] = value
            elif isinstance(init, Iterable) and not isinstance(init, str):
                for item in init:
                    self.add(item)
            else:
                raise TypeError(f"class Data needs a dict or Data to initialize, not {type(init)}.")
    
    def __getitem__(self, key):
        return self._items[key]
    
    def __setitem__(self, key, value):
        if isinstance(value, Column):
            if key == value.name:
                self._items[key] = value
            else:
                raise ValueError(f"The key and the Column.name must be the same")
        else:
            raise TypeError(f"class Data needs Column as value, not a {type(value)}")
    
    def add(self, value):
        if isinstance(value, Column):
            self._items[value.name] = value
        else:
            raise TypeError(f"class Data needs Column as value, not a {type(value)}")
    
    def __delitem__(self, key):
        del self._items[key]
    
    def __iter__(self):
        return iter(self._items)

    def __len__(self):
        return len(self._items)
        
    
    def __repr__(self):
        return f"Data({'\n'.join(column for column in self._items.values())})"
        
    
    def get_array(self, key:str, unit_name:str)->NDArray:
        """
        get a specified numpy.NDArray from Data, under a specified unit.
        """
        column = self[key]
        return column.array * column.unit.convert(unit_name)

    def get_headers(self)->set[str]:
        return set(self)
    


class Model(Protocol):
    quantities: dict[str,dict]
    data_x: NDArray

    def residual(self, fit_x):
        ...
    def forward_factory(self, para_x):
        ...

@runtime_checkable
class HasJacobian(Protocol):
    def jacobian(self, fit_x):
        ...
@runtime_checkable
class HasInitialGuess(Protocol):
    def initial_guess(self):
        ...
@runtime_checkable
class HasTransfrom(Protocol):
    def from_physics(self, para_x:tuple):
        ...
    def to_physics_set(self, fit_x:tuple)->set[tuple]:
        ...



def unit_register(physical_quantity:str, unit_name:str, multiplier:float=1, unicode_name:Optional[str]=None):
    if unicode_name is None:
        unicode_name = unit_name
    registered_units.setdefault(physical_quantity, {})[unit_name] = Unit(
            physical_quantity=physical_quantity,
            name = unit_name,
            multiplier= multiplier,
            unicode_name=unicode_name,
        )

def model_register(model:Model):
    registered_models.setdefault(frozenset(model.quantities), []).append(model)
    return model