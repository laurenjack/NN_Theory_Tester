import numpy as np

class DataSet:
    """
    Individual data set, even for the same problem, the training/validation/test set will be distinct instances
    of this object
    """

    def __init__(self, x, y, is_adversary):
        self.x = x
        self.y = y
        self.is_adversary = is_adversary

    def __str__(self):
        return str(self.x)+'\n\n'+str(self.y)+'\n\n'+str(self.is_adversary)


def combine(*data_sets):
    xs = []
    ys = []
    is_adv = []
    for ds in data_sets:
        xs.append(ds.x)
        ys.append(ds.y)
        is_adv.append(ds.is_adversary)
    return DataSet(np.concatenate(xs), np.concatenate(ys), np.concatenate(is_adv))

