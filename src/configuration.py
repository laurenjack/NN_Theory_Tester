import math
import tensorflow as tf


conf = None  # singleton reference


def get_configuration():
    global conf
    if not conf:
        conf = RbfSoftmaxConfiguration()
    return conf


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
        # The first fields are machine specific and should really be passed in as program args.
        # A parameter to specify where to load the CIFAR10 training and test sets
        self.data_dir = '/home/laurenjack/models/cifar-data'
        # (optional) Directory for saving models, the model will be stored in different sub directories based on
        # different combinations of is_resnet and is_rbf
        self.model_save_dir = '/home/laurenjack/models' # '/home/laurenjack/models'  # '/Users/jack/models'

        # Run an experiment on an NN, or on a toy rbf problem to get gradients right.
        self.is_net = True
        # Compute all_ops in training, rather than just the training operation.
        self.debug_ops = True
        # Specify whether you want to train a convolutional resnet or a standard feed-forward net. This will also
        # inadvertently specify whether CIFAR-10 (True) or MNIST (FALSE) is used for training.
        self.is_resnet = False
        # Use an rbf-softmax at the end of this network, or a softmax (vanilla softmax) end.
        self.is_rbf = True
        # If this is more than 1, train multiple networks and compare their transferability.
        self.n_networks = 1
        # This specifies that the application should try load a network from the location specified by model_save_dir,
        # (and sub directories which are specified by is_resnet and is_rbf)
        self.do_train = True
        # Only does something if is_resnet is False. Then, if is_artificial_data is True, the network will train on
        # the problem specified in artificial_problem.py TODO(Jack) refactor artificial problem name/location
        self.is_artificial_data = False
        # Specify two specific classes, load all examples of just those classes, rather than the whole data set.
        self.just_these_classes = None

        # Network and rbf params
        # The number of dimensions in z space (i.e. the number of neurons at the layer pre softmax / rbf-softmax)
        self.d = 100
        # Rbf Constant c. The scaling factor applied to every rbf value before the softmax is applied
        self.rbf_c = 4.0
        # The initialisation variance of each z_bar value
        self.z_bar_init_sd = 3.0
        # The initial value every tau variable has
        self.tau_init = 1.0
        # Epsilon value to offset dividing by zero when normalizing gradient
        self.norm_epsilon = 10 ** (-8)

        # Feed-forward specific parameters
        # List[int] - where the number of elements corresponds to the number of hidden layers BEFORE the z space.
        # (The z-space is the last hidden layer) The elements themselves are the size of each hidden layer
        self.hidden_sizes = [784, 500]

        # Resnet specific parameters, see: https://arxiv.org/abs/1512.03385
        # The number of filters in the first layer (the layer that moves over the image)
        self.num_filter_first = 32
        # List[int] - The number of stacks (number of elements) and how many filters are in each layer of the stack
        # (the elements themselves). All stacks other than the first stack reduce the width (and height of the filters).
        # Incidentally the number of stacks less 1 specifies how many times the size of the filters drops by a factor
        # of 2. e.g. 32 * 32 -> 16 * 16  -> 8 * 8. Therefore, a config exception will be thrown if the number of
        # elements in this list - 1 is greater than the powers of 2 that factor into the images width.
        self.num_filter_for_stack = [32, 64, 128]
        # List[int] - The number of blocks per stack. Must be the same length as num_filter_for_stack
        self.num_blocks_per_stack = [5, 5, 5]
        # Batch normalisation parameters, resnet
        self.bn_decay = 0.9997
        self.bn_epsilon = 0.001
        # Weight decay parameters, resnet
        self.weight_decay = 0.0002
        self.wd_decay_rate = 0.2
        self.decay_epochs = 100
        # If is_per_filter_fc is true, then instead of using the standard global average pooling post the
        # convolutional layers, use a fully connected layer per filter (see opeartion.py for details) Setting this to
        # True requires that d is a multiple of the last conv layer's number of activations per filter. Consequently
        # an exception will be thrown if d != k * last_filter_width ** 2.0
        self.is_per_filter_fc = True

        # Training parameters
        # Batch size
        self.m = 128
        # Learning Rate, and experimental multipliers on that learning rate for rbf components
        self.lr = 0.01 # 0.001  # * float(self.d) ** 0.5 #0.001 # 0.00001
        self.z_bar_lr_increase_factor = 0.0 # float(self.d)  # ** 0.5
        self.tau_lr_increase_factor = 0.0  # 0.01 / self.lr  #* 3.0 #500.0 # + float(self.d) ** 0.5
        self.epochs = 30
        # The epochs we should decrease the learning rate by decrease_lr_factor
        self.decrease_lr_points = [30, 42, 53]
        self.decrease_lr_factor = 0.01
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
        self.adversarial_ss = 5
        # The number of epochs we make continual perturbations to an adversary
        self.adversarial_epochs = 100
        # Change the image such that the class at index 0 should be perturb into adversary of the class at index 1.
        # If you set the this parameter to None if you would like to use the two closest classes instead, in terms of
        # the unweighted distance between their z_bar centres. (This will only work for rbf networks).
        self.class_to_adversary_class = (0, 7)
        # The arbitrary threshold used to consider when an adverserial example is convincing. This is used by the
        # transferability test to indicate if an example is convincing.
        self.convincing_threshold = 0.7

        # Reporting params (see reporter module)
        # If True, will print the rbf parameters z (valdiation set), z_bar, and tau, only works if is_rbf is True
        self.print_rbf_params = False
        # Report on adversarial examples for the given network
        self.show_adversaries = True
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
        self.n = 1
        self.num_class = 1
        self.num_runs = 1
        self.rbf_only_out_dir = '/home/laurenjack/test_rbf' + str(self.d)  # '/Users/jack/tf_runs/test_rbf5'
        self.show_animation = True
        self.animation_ss = 500
        self.animation_interval = 500
        self.repeat_animation = True
        self.incorr_report_limit = 3
        self.num_duds = 0
        self.do_useless_dimensions = False


def validate(conf, data_set):
    """Validate the configuration is consistent, as pertaining to network training and construction.

    By no means does this cover all inconsistencies, just the ones that aren't obvious to the user.

    Args:
        conf: see configuration.RbfSoftmaxConfiguration
        data_set: The data set the network specified by conf will be trained on.

    Raises: ConfigurationException - when the set of configurations for network construction and training is
    inconsistent and unresolvable.
    """
    if conf.is_resnet:
        # Validate the number of stacks isn't too big
        image_width = data_set.image_width
        num_stacks = len(conf.num_filter_for_stack)
        num_width_reductions = num_stacks - 1
        if image_width % (2 ** num_width_reductions) != 0:
            raise ConfigurationException("\nToo many stacks, i.e. num_filter_for_stack is too long. Recall that\n"
                                         "each stack reduces the filter width by a half, this is not possible with\n"
                                         "an image_width of {} and {} stacks\n".format(image_width, num_stacks))

        # Validate that there is a block size specified for every stack.
        if num_stacks != len(conf.num_blocks_per_stack):
            raise ConfigurationException("\nThe number of filters per stack: {} specifies there are {} stacks,\n"
                                         "so the number of blocks per stack: {} has an incorrect length.\n"
                                         .format(conf.num_filter_for_stack, num_stacks, conf.num_blocks_per_stack))

        # If per filter fc is used, validate d is a multiple of the last conv layers number of filters
        num_filter = conf.num_filter_for_stack[-1]
        if conf.is_per_filter_fc and conf.d % num_filter != 0:
            raise ConfigurationException("\nWhen using a per-filter-fully-connected layer after the conv filters,\n"
                                         "you must make sure that d is a multiple of the last number of filters\n"
                                         "However, d: {} and num_filter_for_stack has last element {}\n"
                                         .format(conf.d, num_filter))


class ConfigurationException(Exception):
    """Represents a logically inconsistent and unresolvable set of configuration parameters
    """
    pass
