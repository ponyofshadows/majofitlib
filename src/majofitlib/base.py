from __future__ import annotations
from typing import Protocol, runtime_checkable, Callable, Optional, NamedTuple
from numpy.typing import NDArray
from collections.abc import MutableMapping
import numpy as np


registered_quantities = {}
registered_models = {}

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


class ArrayWithUint(NamedTuple):
    array: NDArray
    unit_name: str


class Data(MutableMapping):
    __slots__ = ("_items")
    def __init__(self, init:Optional[dict|Data]=None):
        self._items = {
            "X":{}, "Y":{}, "PARA":{}
                        }
        if init:
            if isinstance(init, Data):
                self._items = dict(init._items)
            elif isinstance(init, dict):
                for key, value in init.items():
                    self[key] = value
            else:
                raise TypeError(f"class Data needs a dict to initialize, not {type(init)}.")
    
    def __getitem__(self, key):
        array_type, physical_quantity = key
        return self._items[array_type][physical_quantity]
    
    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            try:
                array_type, physical_quantity = key[0:2]
            except ValueError:
                raise KeyError("The key should be a tuple of an array type and a physical quantity name." \
                    "For example: 'X, magnetic field'; 'PARA, charge'")
        else:
            raise TypeError(f"The key should be a tuple of an array type name and a physical quantity name', not {type(key)}")
        
        if isinstance(value, ArrayWithUint):
            array_with_unit = value
        elif isinstance(value, tuple):
            try:
                array_value, unit_name = value[0:2]
            except ValueError:
                raise ValueError(f"The value should be a tuple of an array-like object and a unit string'")
            array_with_unit = ArrayWithUint(np.asarray(array_value), unit_name)
        else:
            raise TypeError(f"The value should be a tuple of an array-like object and a unit string', not {type(value)}")

        try:
            self._items[array_type][physical_quantity] = array_with_unit
        except KeyError:
            raise TypeError(f"{array_type} is a wrong array type. Try 'X', 'Y' or 'PARA'.")
    
    def __delitem__(self, key):
        array_type, physical_quantity = key
        del self._items[array_type][physical_quantity]
    
    def __iter__(self):
        def key():
            for array_type in self._items:
                for quantity in self._items[array_type]:
                    yield (array_type, quantity)
        return key()

    def __len__(self):
        if self._items["X"]:
            return len(next(iter(self._items["X"])))
        else:
            return 0
        
    
    def __repr__(self):
        return "Data()" + "\n".join([f"{key[0]}: {key[1]}={self[key]}" for key in self])
    
    def get_array(self,quantities:dict[str,dict], key:tuple[str], unit_name:str)->NDArray:
        """
        get a specified numpy.NDArray from Data, under a specified unit.
        """
        array_with_unit = self[key]
        return array_with_unit.array * (quantities[key][unit_name] / array_with_unit.unit)

    def get_headers(self)->set[tuple[str]]:
        return set(self)
    


class Forward(Protocol):
    def __call__(self, x):
        ...

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



def quantity_unit_register(physical_quantity:str, unit_name:str, multiplier:float=1, unicode_name:Optional[str]=None):
    if unicode_name is None:
        unicode_name = unit_name
    registered_quantities.setdefault(physical_quantity, {})[unit_name] = Unit(
            name = unit_name,
            multiplier= multiplier,
            unicode_name=unicode_name,
        )

def model_register(model:Model):
    registered_models.setdefault(frozenset(model.quantities), []).append(model)
    return model