import tensorflow as tf
import numpy as np
import src.variable_creator as vc
import bar_network as bn
import train_experiment as te
import bar_experiment as be

# network/training parameters
num_filters = [3, 2]
epochs = 20
batch_size = 8
lr = 0.1

# Experiment parameters
num_runs = 100

tf.logging.set_verbosity(tf.logging.ERROR)

def run_bar_experiment():
    # Manual Dependency Injection
    trainer = te.ExperimentTrainer(num_runs, epochs, batch_size)
    variable_creator = vc.VariableCreator()
    bar_network = bn.BarNetwork(variable_creator, num_filters, lr)
    training_set = be.training_set()
    validation_set = be.all_value_validation_set()
    adverserial_set = be.adverserial_set()

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    all_correct_flags = []
    for run_number in xrange(num_runs):
        all_correct = trainer.train(sess, bar_network, training_set, validation_set, adverserial_set, run_number)
        all_correct_flags.append(all_correct)
        # Re-init training variables
        tf.global_variables_initializer().run()

    # Report on the number of each type of example which was all correct
    print 'How many times were all the examples correct?'
    train_all_correct, val_all_correct, adv_all_correct = zip(*all_correct_flags)
    _report_fraction_all_correct('Train', train_all_correct)
    _report_fraction_all_correct('Validation', val_all_correct)
    _report_fraction_all_correct('Adverserial', adv_all_correct)

def _report_fraction_all_correct(name, all_correct_flags):
    # TODO refactor into reporting framework
    all_correct_flags = np.array(all_correct_flags, dtype=np.bool)
    n = all_correct_flags.shape[0]
    num_all_correct = np.sum(all_correct_flags.astype(dtype=np.int32))
    print '{name}: {num_correct} / {n}'.format(name=name, num_correct=num_all_correct, n=n)



if __name__ == '__main__':
    run_bar_experiment()