import numpy as np
import visualisation
import configuration

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


def roc_curve(X, Y, network_runner, conf):
    # TODO Current function assumes perfectly balanced classes
    Y = Y.astype(np.int32)
    probabilities = network_runner.probabilities(X, Y)
    n = probabilities.shape[0]
    class_wise_tprs = []
    class_wise_fprs = []
    for k in xrange(conf.num_class):
        actual_k_indicies = np.argwhere(Y == k)[:, 0]
        not_k_indicies = np.argwhere(Y != k)[:, 0]
        probability_of_k = probabilities[:, k]
        prob_of_k_for_actual_ks = probability_of_k[actual_k_indicies]
        ranked_prob_of_k_for_actual_ks = -np.sort(-prob_of_k_for_actual_ks)
        not_k_probabilities = probability_of_k[not_k_indicies]

        samples_per_k = actual_k_indicies.shape[0]
        tprs = []
        fprs = []
        for i in xrange(0, samples_per_k):
            k_recalled = i+1
            thresh = ranked_prob_of_k_for_actual_ks[i]
            falsely_recalled = np.sum(np.greater_equal(not_k_probabilities, thresh).astype(np.int32))
            tpr = float(k_recalled) / float(samples_per_k)
            fpr = float(falsely_recalled) / float(n - samples_per_k)
            tprs.append(tpr)
            fprs.append(fpr)
        # Weight the tprs and fprs accordingly
        weight = float(samples_per_k) / float(n)
        tprs = np.array(tprs) * weight
        fprs = np.array(fprs) * weight
        class_wise_tprs.append(tprs)
        class_wise_fprs.append(fprs)

    final_tprs = _sum_arrays(class_wise_tprs)
    final_fprs = _sum_arrays(class_wise_fprs)
    return final_tprs, final_fprs


def _sum_arrays(arrays):
    base = np.zeros(arrays[0].shape, dtype=np.float32)
    for a in arrays:
        base += a
    return base


def write_csv(X, Y, network_runner):
    """ETL pipline taking the data regarding an rbf networks predictions
    and writing a relational model to a set of csv files"""
    point_stat, dimension_stat = extract_and_transform(X, Y, network_runner)
    np.savetxt("prediction_output/point_stat.csv", point_stat.transpose(), fmt="%.2f", delimiter=",")
    np.savetxt("prediction_output/dimension_stat.csv", dimension_stat.transpose(), fmt="%.2f", delimiter=",")




