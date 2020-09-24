from abc import ABCMeta


class IAugmentation(metaclass=ABCMeta):
    @staticmethod
    def set_successor(successor):
        """Set the next handler"""

    @staticmethod
    def handle(image):
        """Handle data"""


class AugmentationPipeline:

    def __init__(self):
        pass
