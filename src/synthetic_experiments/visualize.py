import matplotlib.pyplot as plt
import numpy as np

def show_net_over_time(w, b, w_soft):
    #Get parameters over time in a form each parmater can be observed individually
    w = np.array(w).transpose([1, 2, 0])
    b = np.array(b).transpose()
    w_soft = np.array(w_soft).transpose([2, 1, 0])

    plt.subplot(121)
    plt.plot(w[0, 0])
    plt.plot(w[1, 0])
    plt.plot(b[0])

    plt.subplot(122)
    plt.plot(w_soft[0, 0])
    plt.plot(w_soft[1, 0])
    plt.show()


