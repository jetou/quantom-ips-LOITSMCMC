from abc import ABC, abstractmethod


class Algorithm(ABC):
    """
    An Abstract Base Class for defining the structure of an Algorithm.
    Subclasses must implement the 'match' and 'apply' methods.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def match(self, tag):
        """
        Abstract method to determine if the algorithm should be applied.
        """
        raise NotImplementedError

    @abstractmethod
    def apply(self, opt, tag):
        """
        Abstract method containing the core logic of the algorithm.
        """
        raise NotImplementedError
