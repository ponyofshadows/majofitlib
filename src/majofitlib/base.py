from typing import Protocol, runtime_checkable, Callable

class ModelStrategy(Protocol):
    @staticmethod
    def model(self, x)->Callable:
        ...
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
    def to_physics(self, x:tuple)->tuple|set[tuple]:
        ...
