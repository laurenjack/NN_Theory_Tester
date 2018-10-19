import math
import tensorflow as tf


class RbfSoftmaxConfiguration:  # TODO(Jack) set seed somewhere for np and tf
    """
    Responsible for configuring a network for the rbf softmax experiment.

    The purpose of this class is to allow convenient switching between various forms of rbf-softmax models and various
    forms of standard neural networks. As I've tried more variants, this configuration has outgrown itself and become a
    bit of a brittle beast with some annoying dependencies between config fields. Rather than write a sophisticated
    configuration scheme, I've simply tried to mitigate this with extensive field descriptions and a logical ordering,
    where in a group of fields, those which are higher level, and impact the other fields are specified first, so that
    the annoying dependencies can be addressed sequentially.

    The current implementation is a singleton, where all properties are static. Although not ideal, I opted for a
    singleton class rather than a set of static fields as this decouples the modules in this project from any one set
    of static configurations. Furthermore, using a class allows for switching to dynamic configuration (say for
    hyper-parameter search) without breaking the contract used by all modules in this project.
    """

    def __init__(self):
        # TODO(Jack) Add as command line arguments
        # The first fields are machine specific and should really be passed in as program args
        # Directory for saving models, separate sub-directories will be created based is_rbf
        self.model_save_dir = '/home/laurenjack/models'
        # parameter to specify where to load the CIFAR training and test sets
        self.data_dir = '/home/laurenjack/models/cifar-data'

        # Run an experiment on an NN, or on a toy rbf problem to get gradients right.
        self.is_net = True
        # Compute all_ops in training, rather than just the training operation.
        self.debug_ops = True
        # Specify whether you want to train a convolutional resnet or a standard feed-forward net. This will also
        # inadvertently specify whether CIFAR-10 (True) or MNIST (FALSE) is used for training.
        self.is_resnet = True
        # Use an rbf-softmax at the end of this network, or a softmax (vanilla softmax) end.
        self.is_rbf = True
        # If is_resent is true AND do_train is False, this specifies that the application should try load a network
        # from the location specified by model_save_dir. If is_resent is True true AND do_train is True the application
        # will train a new network and save it in model_save_dir. However if is_resnet is False, the feed-forward net
        # will just go ahead and train, without saving the model (why bother its quick)
        self.do_train = True
        # Only does something if is_resnet is False. Then, if is_artificial_data is True, the network will train on
        # the problem specified in artificial_problem.py TODO(Jack) refactor artificial problem name/location
        self.is_artificial_data = False

        # TODO(Jack) refactor these data set specific parameters, they can be inferred from the data set itself
        self.n = 1
        self.num_class = 10
        self.image_width = 32
        self.image_depth = 3

        # Network and rbf params
        # The number of dimensions in z space (i.e. the number of neurons at the layer pre softmax / rbf-softmax)
        self.d = 4096
        # Rbf Constant c. The scaling factor applied to every rbf value before the softmax is applied
        self.rbf_c = 4.0
        # The initialisation variance of each z_bar value
        self.z_bar_init_sd = 3.0
        # The initial value every tau variable has
        self.tau_init = 0.5
        # Epsilon value to offset dividing by zero when normalizing gradient
        self.norm_epsilon = 10 ** (-10)

        # Feed-forward specific parameters
        # List[int] - where the number of elements corresponds to the number of hidden layers BEFORE the z space.
        # (The z-space is the last hidden layer) The elements themselves are the size of each hidden layer
        self.hidden_sizes = [784]

        # Resnet specific parameters, see: https://arxiv.org/abs/1512.03385
        # The number of filters in the first layer (the layer that moves over the image)
        self.num_filter_first = 16
        # List[int] - The number of stacks (number of elements) and how many filters are in each layer of the stack
        # (the elements themselves)
        self.num_filter_for_stack = [16, 32, 64]
        # List[int] - The number of blocks per stack. Must be the same length as num_filter_for_stack
        self.num_blocks_per_stack = [5, 5, 5]
        self.kernel_width = 3
        self.kernel_stride = 1
        # The kernel stride when moving from one stack to the next thinner stack.
        self.stack_entry_kernel_stride = 2
        # Batch normalisation parameters, resnet
        self.bn_decay = 0.9997
        self.bn_epsilon = 0.001
        # Weight decay parameters, resnet
        self.weight_decay = 0.0002
        self.wd_decay_rate = 0.2
        self.decay_epochs = 100

        # Training parameters
        # Batch size
        self.m = 128
        # Learning Rate, and experimental multipliers on that learning rate for rbf components
        self.lr = 0.003  # * float(self.d) ** 0.5 #0.001 # 0.00001
        self.z_bar_lr_increase_factor = float(self.d)  # ** 0.5
        self.tau_lr_increase_factor = 0.0  # 0.01 / self.lr  #* 3.0 #500.0 # + float(self.d) ** 0.5
        self.epochs = 10
        # The epochs we should decrease the learning rate by decrease_lr_factor
        self.decrease_lr_points = [80, 120]
        self.decrease_lr_factor = 0.01
        # The optimizer to use for training the network
        # TODO(Jack) currently the code is coupled to a momentum optimizer
        self.optimizer = tf.train.MomentumOptimizer
        # The target precision of the z instances
        self.target_precision = 1.0
        # The sample size when used when reporting accurarcy after each epoch of training
        self.accuracy_ss = 1000

        # Shortest point finder params (see shortest_point_finder module)
        self.spf_lr = 0.01
        self.spf_lmda = math.pi / 1440
        self.spf_epochs = 200
        self.spf_animation_interval = 50

        # Adversarial params (see adversarial module)
        # Also see - Fast Gradient Sign Method https://arxiv.org/abs/1412.6572
        # The size of the perturbation, at each step
        self.adversarial_epsilon = 0.01
        # The number of adversarial examples to generation
        self.adversarial_ss = 10
        # The number of epochs we make continual perturbations to an adversary
        self.adversarial_epochs = 100
        # Change the image such that the class at index 0 should be perturb into adversary of the class at index 1.
        # If you set the this parameter to None if you would like to use the two closest classes instead, in terms of
        # the unweighted distance between their z_bar centres. (This will only work for rbf networks).
        self.class_to_adversary_class = (3, 5)

        # Reporting params (see reporter module)
        # If True, will print the rbf parameters z (valdiation set), z_bar, and tau, only works if is_rbf is True
        self.print_rbf_params = False
        # Report on adversarial examples for the given network
        self.show_adversaries = False
        # Show an ROC curve for the validation set
        self.show_roc = False
        # Show the 5 most incorrect examples - where most incorrect means, the 5 incorrectly classified validation
        # examples with the highest prediction probabilities
        self.show_really_incorrect = False
        self.top_k_incorrect = 5
        # Write a table data structure for analytics to a csv file (see prediction_analytics module)
        self.write_csv = False
        self.show_z_stats = False

        # TODO(Jack) handle when refactoring artifical_problem.py)
        self.artificial_in_dim = 2

        # Rbf only parameters, for when is_net is False and we are only training z variables directly (rather than
        # a network TODO(Jack) lot's of deprecated stuff to get rid of here
        self.num_runs = 1
        self.rbf_only_out_dir = '/home/laurenjack/test_rbf' + str(self.d)  # '/Users/jack/tf_runs/test_rbf5'
        self.show_animation = True
        self.animation_ss = 500
        self.animation_interval = 500
        self.repeat_animation = True
        self.incorr_report_limit = 3
        self.num_duds = 0
        self.do_useless_dimensions = False


conf = None  # singleton reference


def get_configuration():
    global conf
    if not conf:
        conf = RbfSoftmaxConfiguration()
    return conf
