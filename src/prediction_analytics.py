import numpy as np
import visualisation

def extract_and_transform(X, Y, network_runner):
    """Extract and transform the data regarding an rbf networks predictions, to a relational model"""
    # Load the network data
    n = X.shape[0]
    correct, incorrect, correct_inds, incorr_inds = network_runner.all_correct_incorrect(X, Y)
    new_ordering = np.concatenate([correct_inds, incorr_inds])
    z, z_bar, tau = network_runner.report_rbf_params(X[new_ordering], Y[new_ordering])

    #Transform, n level
    num_correct = correct.a.shape[0]
    num_incorrect = n - num_correct
    point_ids = np.arange(n)
    predicted = np.concatenate([correct.prediction, incorrect.prediction])
    prediction_prob = np.concatenate([correct.prediction_prob(), incorrect.prediction_prob()])
    is_correct = np.concatenate([np.ones(num_correct, dtype=np.bool), np.zeros(num_incorrect, dtype=np.bool)])
    point_stat = np.array([point_ids, predicted, prediction_prob, is_correct], dtype=object)

    #Transform n * d level
    d, K = z_bar.shape
    point_ids_nd = np.array([point_ids] * d).flatten('F')
    dim_ids = np.array([np.arange(d)] * n).flatten()
    z_diff = z.reshape((n, d, 1)) - z_bar.reshape(1, d, K)
    z_diff_k_first = z_diff.transpose([0, 2, 1])
    selected_z_diff = z_diff_k_first[point_ids, predicted]
    selected_z_diff = selected_z_diff.flatten()
    selected_tau = tau.transpose()[predicted].flatten()
    selected_z = z.flatten()
    selected_z_bar = z_bar.transpose()[predicted].flatten()
    dimension_stat = np.array([point_ids_nd, dim_ids, selected_z_diff, selected_tau, selected_z, selected_z_bar], dtype=object)
    return point_stat, dimension_stat

def roc_curve(X, Y, network_runner):
    # TODO | This is a lazy and technically incorrect implementation, would need to go class wise to get the true false
    # TODO | positive rates.
    n = X.shape[0]
    point_stat, _ = extract_and_transform(X, Y, network_runner)
    point_stat = point_stat.transpose()
    thresh_variant = np.arange(0, 1.0, 0.01)
    tps = []
    fps = []
    # tps.append(1.0)
    # fps.append(1.0)
    for i in xrange(thresh_variant.shape[0]):
        t = thresh_variant[i]
        tp_condition = np.logical_and(point_stat[:, 3].astype(np.bool), point_stat[:, 2] > t)
        tp = point_stat[tp_condition].shape[0] / float(n)
        fp_condition = np.logical_and(np.logical_not(point_stat[:, 3].astype(np.bool)), point_stat[:, 2] > t)
        fp = point_stat[fp_condition].shape[0] / float(n)
        tps.append(tp)
        fps.append(fp)
    visualisation.plot('ROC curve', fps, tps)





def write_csv(X, Y, network_runner):
    """ETL pipline taking the data regarding an rbf networks predictions
    and writing a relational model to a set of csv files"""
    point_stat, dimension_stat = extract_and_transform(X, Y, network_runner)
    np.savetxt("prediction_output/point_stat.csv", point_stat.transpose(), fmt="%.2f", delimiter=",")
    np.savetxt("prediction_output/dimension_stat.csv", dimension_stat.transpose(), fmt="%.2f", delimiter=",")




