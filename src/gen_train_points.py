import numpy as np
from configuration import conf

def gen_train_labels():
    n = conf.n
    d = conf.d
    num_class = conf.num_class
    num_reps = n // num_class + 1
    #balanced classes
    y = np.array(range(num_class) * num_reps)
    y = y[:n]
    #np.random.shuffle(y)
    return y

