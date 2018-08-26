import numpy as np
import visualisation
from configuration import conf

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
        tprs = np.array(tprs)
        fprs = np.array(fprs)
        class_wise_tprs.append(tprs)
        class_wise_fprs.append(fprs)


    final_tprs = combine(class_wise_tprs)
    final_fprs = combine(class_wise_fprs)

    return final_tprs, final_fprs


def combine(prs):
    k = len(prs)
    min_ss = _find_min_length(prs)
    mean_pr = np.zeros(min_ss)
    for i in xrange(min_ss):
        proportion_included = float(min_ss) / float(i+1)
        for pr in prs:
            floor = int(pr.shape[0] / proportion_included) - 1
            mean_pr[i] += pr[floor]
    return mean_pr / k





def _find_min_length(arrays):
    return min([arr.shape[0] for arr in arrays])

def not_roc_curve(X, Y, network_runner):
    """It would be incorrect to call this an ROC curve, as this is not for n-ary classification and is not computed
    in the same way. Instead we look at the number of correct guesses relative to the set size, agaisnt the number
    of incorrect guesses relative to the sample size. That is, there is no weighting according to class like their
    would be with tpr or fpr."""
    # TODO Figure out what this curve is really called or give it a new name if it's a new thing
    n = X.shape[0]
    corr, incorr, correct_inds, incorr_inds = network_runner.all_correct_incorrect(X, Y)
    probs_corr = corr.a
    probs_incorr = incorr.a

    # Rank the correct predictions by their probability
    ranked_prob_correct = -np.sort(-probs_corr)

    correct_at_each_recall = np.zeros(n)
    incorrect_at_each_recall = np.zeros(n)
    for i in xrange(0, n):
        num_recalled = i+1
        thresh = probs_corr[i]
        falsely_recalled = np.sum(np.greater_equal(probs_incorr, thresh).astype(np.int32))
        correct_at_each_recall[i] = float(num_recalled)
        incorrect_at_each_recall[i] = float(falsely_recalled)

    Y = Y.astype(np.int32)
    probabilities = network_runner.probabilities(X, Y)

    class_wise_tprs = []
    class_wise_fprs = []
    prob_of_ks_for_actual_ks = []
    not_ks_probabilities = []
    for k in xrange(conf.num_class):
        actual_k_indicies = np.argwhere(Y == k)[:, 0]
        not_k_indicies = np.argwhere(Y != k)[:, 0]
        probability_of_k = probabilities[:, k]
        prob_of_k_for_actual_k = probability_of_k[actual_k_indicies]
        prob_of_ks_for_actual_ks.append(prob_of_k_for_actual_k)
        not_ks_probabilities.append(probability_of_k[not_k_indicies])

    prob_of_ks_for_actual_ks = np.concatenate(prob_of_ks_for_actual_ks)
    not_ks_probabilities = np.concatenate(not_ks_probabilities)
    # Rank the actually correct k's by the size of their probabilities, highest first
    ranked_prob_of_k_for_actual_k = -np.sort(-prob_of_ks_for_actual_ks)



    return correct_at_each_recall / float(n), incorrect_at_each_recall / float(n)


def write_csv(X, Y, network_runner):
    """ETL pipline taking the data regarding an rbf networks predictions
    and writing a relational model to a set of csv files"""
    point_stat, dimension_stat = extract_and_transform(X, Y, network_runner)
    np.savetxt("prediction_output/point_stat.csv", point_stat.transpose(), fmt="%.2f", delimiter=",")
    np.savetxt("prediction_output/dimension_stat.csv", dimension_stat.transpose(), fmt="%.2f", delimiter=",")




