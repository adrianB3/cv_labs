from abc import abstractmethod, ABC


class Data:
    """Base data class."""
    def __init__(self, **kwargs):
        self.data = kwargs


class Augmentation(ABC):
    """Base augmentation class."""

    @abstractmethod
    def process(self, data: Data):
        pass
