conf = None  # singleton reference


def get_configuration():
    global conf
    if not conf:
        conf = ConfigurationNetwork()
    return conf


class ConfigurationNetwork(object):
    """Configuration specifically for building a neural network in the context of activation distribution estimation.
    """

    def __init__(self):
        self.data_dir = None
        self.model_save_dir = None
        self.dataset_name = None
        self.debug_ops = False
        self.is_resnet = False
        self.is_rbf = False
        self.n_networks = 1
        self.do_train = True
        self.use_orthogonality_filters = False
        self.is_artificial_data = False
        self.just_these_classes = None
        self.d = 100
        self.hidden_sizes = []
        self.m = 128
        self.lr = 0.02
        self.epochs = 20
        self.decrease_lr_points = []
        self.decrease_lr_factor = 0.01
        self.adversarial_ss = 10
        self.adversarial_epochs = 5
        self.adversarial_epsilon = 0.01
        self.accuracy_ss = 1000
        self.class_to_adversary_class = (3, 5)
        self.show_adversaries = False
        self.show_node_distributions = False
        self.number_of_node_distributions = 3
        self.write_csv = False
        self.show_roc = False
        self.show_really_incorrect = False
