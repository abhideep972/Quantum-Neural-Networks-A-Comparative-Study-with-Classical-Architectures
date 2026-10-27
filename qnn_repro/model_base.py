from abc import ABC, abstractmethod

class Model(ABC):
    def __init__(self):
        self.thetamin = 0.0
        self.thetamax = 1.0

    @abstractmethod
    def forward(self, *args, **kwargs): ...
    @abstractmethod
    def get_gradient(self, *args, **kwargs): ...
    @abstractmethod
    def get_fisher(self, *args, **kwargs): ...
