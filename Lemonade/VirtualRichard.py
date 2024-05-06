from Typing import Callable


class VirtualRichard:
    def __init__(self, ranking_function: Callable = None):
        self.ranking_function = ranking_function

    def rank(self, data):
        if self.ranking_function is None:
            raise ValueError("No ranking function provided")
        data_ranks = [self.ranking_function(seg) for seg in data] # ranks of each segmentation according its rank
        return data_ranks

    def resegment(self, data):
        pass



