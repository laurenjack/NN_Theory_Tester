import tensorflow as tf
from animator import *
import train_rbf
from reporter import *
from network_factory import *
import rbf
import configuration
conf = configuration.get_conf()


def run_network(conf):
    network_runner,data_set = create_and_train_network()

    # Report on the results
    report_single_network(network_runner, data_set, conf)
    # report_with_adversaries_from_second(network_runner1, network_runner2, data_set, conf)

def run_rbf_test(conf):
    total_correct = 0
    for i in xrange(conf.num_runs):
        train_result = train_rbf.train(conf)
        train_result.report_incorrect()
        num_correct = train_result.num_correct
        print float(num_correct) / float(conf.n) * 100
        total_correct += num_correct
        z_bar = train_result.final_z_bar
        tau = train_result.final_tau
        sp_z_list, Cs, rbfs, z_bar_pair, tau_pair = find_shortest_point(conf, z_bar, tau)
        print Cs
        print rbfs
    print ""
    tpc = float(total_correct) / float(conf.n * conf.num_runs) * 100
    print "Total Percentage Correct: " + str(tpc)

    # Take the last a and evaluate the percentage of correctly classified points
    if conf.num_runs == 1 and conf.d == 2:
        animate(train_result, conf)
        animate_spf(z_bar_pair, tau_pair, sp_z_list, conf)

if __name__ == '__main__':
    conf = configuration.get_conf()
    #run_rbf_test(conf)
    run_network(conf)



