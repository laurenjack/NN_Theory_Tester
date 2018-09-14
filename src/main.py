import tensorflow as tf
from animator import *
import train_rbf
from reporter import *
import network_factory
import rbf
from configuration import conf


def run_network():
    network_runner, data_set = network_factory.create_and_train_network()

    # Report on the results
    report_single_network(network_runner, data_set)
    # report_with_adversaries_from_second(network_runner1, network_runner2, data_set, conf)

def run_rbf_test():
    total_correct = 0
    for i in xrange(conf.num_runs):
        train_result = train_rbf.train()
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
    if conf.num_runs == 1: # and conf.d == 2:
        animate(train_result.get())
        #animate_spf(z_bar_pair, tau_pair, sp_z_list)

if __name__ == '__main__':
    if conf.is_net:
        run_network()
    else:
        run_rbf_test()



