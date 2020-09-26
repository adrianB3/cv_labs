from abc import abstractmethod, ABC
from typing import Any

from augmentation.data_types import Data


class IAugmentationHandler(ABC):

    @abstractmethod
    def set_next(self, handler):
        """Set the next handler"""
        pass

    @abstractmethod
    def handle(self, request):
        """Handle data"""
        pass


class BaseAugmentationHandler(IAugmentationHandler):
    _next_handler: IAugmentationHandler = None

    def set_next(self, handler: IAugmentationHandler) -> IAugmentationHandler:
        self._next_handler = handler
        # allow easy chaining
        return handler

    def handle(self, request: Any) -> Any:
        if self._next_handler:
            return self._next_handler.handle(request)
        return None


class Pipeline:
    def __init__(self):
        self.augmentations = []

    def add_augmentation(self, augmentation):
        self.augmentations.append(augmentation)
        return self  # allow chaining

    def execute(self, data: Data):
        for aug in self.augmentations:
            aug.process(data)
            data.data['applied_augmentations'].append(str(aug.__class__.__name__))
