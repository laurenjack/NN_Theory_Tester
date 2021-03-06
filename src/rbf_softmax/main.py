import src.rbf_softmax.network_factory
import src.rbf_softmax.train_rbf
import src.reporter_factory
from src.rbf_softmax import animator, configuration


def run_network(conf):
    """"Create, train and reports on a neural network architecture according to conf.

    Args:
        conf: configuration.RbfSoftmaxConfiguration
    """
    network_runner, data_set, training_results = src.rbf_softmax.network_factory.create_and_train_network(conf)
    reporter = src.reporter_factory.create_reporter(data_set)
    reporter.report_single_network(network_runner, data_set, training_results)


def run_networks_and_report_transferability(conf):
    """Train a series of networks, then test the transferability of adversarial attacks.

    Args:
        conf: configuration.RbfSoftmaxConfiguration
    """
    network_runners, data_set = src.rbf_softmax.network_factory.create_and_train_n_networks(conf)
    reporter = src.reporter_factory.create_reporter(data_set)
    reporter.report_with_adversaries_from_first(network_runners, data_set, conf.convincing_threshold)


def run_rbf_test(conf):  # TODO(Jack) sort this rbf only code
    total_correct = 0
    for i in xrange(conf.num_runs):
        train_result = src.rbf_softmax.train_rbf.train()
        train_result.report_incorrect()
        num_correct = train_result.num_correct
        print float(num_correct) / float(conf.n) * 100
        total_correct += num_correct
        z_bar = train_result.final_z_bar
        tau = train_result.final_tau
        #sp_z_list, Cs, rbfs, z_bar_pair, tau_pair, _ = find_shortest_point(z_bar, tau)
        #print Cs
        #print rbfs
    print ""
    tpc = float(total_correct) / float(conf.n * conf.num_runs) * 100
    print "Total Percentage Correct: " + str(tpc)

    # Take the last a and evaluate the percentage of correctly classified points
    if conf.num_runs == 1:  # and conf.d == 2:
        animator.animate(*train_result.get())


def run():
    conf = configuration.get_configuration()
    if conf.is_net:
        if conf.n_networks and conf.n_networks > 1:
            run_networks_and_report_transferability(conf)
        else:
            run_network(conf)
    else:
        run_rbf_test(conf)


if __name__ == '__main__':
    run()



