import numpy as np


def extract_and_transform(X, Y, network_runner):
    """Extract and transform the data regarding an rbf networks predictions, to a relational model"""
    # Load the network data
    n = X.shape[0]
    correct, incorrect, new_ordering = network_runner.all_correct_incorrect(X, Y)
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

def write_csv(X, Y, network_runner):
    """ETL pipline taking the data regarding an rbf networks predictions
    and writing a relational model to a set of csv files"""
    point_stat, dimension_stat = extract_and_transform(X, Y, network_runner)
    np.savetxt("prediction_output/point_stat.csv", point_stat.transpose(), fmt="%.2f", delimiter=",")
    np.savetxt("prediction_output/dimension_stat.csv", dimension_stat.transpose(), fmt="%.2f", delimiter=",")

