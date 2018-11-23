import tensorflow as tf


class Rbf(object):
    """Represents an rbf-softmax graph, which begins at the z-space and ends at the cross entropy loss function.

    Intended be utilised as the end of a neural network, see public methods for details.
    """

    def __init__(self, conf, z_bar_init, tau_init, num_class, network_id):
        # Unpack configuration
        self.num_class = num_class
        self.d = conf.d
        self.rbf_c = conf.rbf_c
        self.norm_epsilon = conf.norm_epsilon
        self.z_bar_lr_increase_factor = conf.z_bar_lr_increase_factor
        self.tau_lr_increase_factor = conf.tau_lr_increase_factor
        self.target_precision = conf.target_precision
        self.network_id = network_id

        # Assign initializers and placeholders
        self.z_bar_init = z_bar_init
        self.tau_init = tau_init
        self.y = tf.placeholder(tf.int32, shape=[None], name="y")
        self.y_hot = tf.one_hot(self.y, self.num_class)
        self.batch_size = tf.placeholder(tf.int32, shape=[], name="batch_size")

        # Necessary state, for removing and re-introducing gradients in backProp
        self.xe_sm_grad = None
        self.tau_quadratic_grad = None
        self.z_grad = None  # TODO(Jack) more elegant debugging strategy
        self.z_bar_grad = None
        # z difference squared, used to scale the tau gradient (see README)
        self.z_difference_squared = None

    def tensors_for_network(self, z):
        """Build an rbf-softmax end, and return the rbf tensors required as part of a larger rbf-softmax network.

        See create_all_ops for details.

        Args:
            z: An m x d floating point tensor, which represents the current batch in the z space.

        Returns:
            A list of tensors, for use in the construction of a rbf-softmax network.
        """
        rbf_ops = self.create_all_ops(z)
        return [rbf_ops.a, rbf_ops.loss, rbf_ops.z, rbf_ops.z_bar, rbf_ops.tau, rbf_ops.rbf]

    def create_all_ops(self, z, batch_indices=None):
        """Constructs the the rbf-softmax section of a graph (e.g. a neural network), and return all relevant tensors.

        This method is responsible for taking the z space and performing all operations between there and the output
        of cross entropy loss function. It initializes the rbf variables, applies the rbf operators to the z_space,
        applies the softmax function to the rbf outputs and finally the loss function. The specific modifications
        to rbf-softmax gradients are applied entirely within (or below) the scope of this method.

        Args:
            z: An m x d floating point tensor, which represents the current batch in the z space. (Or an n x d tensor,
            may be used, where a batch_indices argument is applied)

            batch_indices: optional - Used if z is specified as a n x d tensor. (In practice this is only used in
            direct rbf experiments, where z is a variable rather than the outputs of an NN layer)

        Returns:
            Returns an instance of RbfTensors (see rbf.RbfTensors) which encapsulates all the tensor needed by
            other rbf-softmax components (e.g. those tensors needed by a neural network such as the loss function).
        """
        # Create the rbf parameters
        graph = tf.get_default_graph()
        z, z_tile = self._tile_z(graph, z, batch_indices)
        z_bar, z_bar_tile = self._create_z_bar(graph)
        tau, tau_tile = self._create_tau(graph)

        z_diff = z_tile - z_bar_tile
        self.z_difference_squared = z_diff ** 2.0
        tau_square_tile = tau_tile ** 2.0 #TODO(Jack) do I actually need to tile tau? Can't I just use a matmul?

        zero_the_grad = "zero_the_grad_"+self.network_id

        @tf.RegisterGradient(zero_the_grad)
        def _zero_the_grad(unused_op, grad):
            return tf.zeros(tf.shape(grad))

        # Here the identity function as usual with a gradient pattern, but with an additional purpose. The tensor
        # pointed to by tau_sq_alternate starts a branch to the main loss, the tensor pointed to by z_diff_sq_alternate
        # starts a branch to the tau_loss. On each of these seperate branches we repeat some of the operations because
        # we don't want to pollute the gradients of the main loss with the gradients of the tau loss or vice versa,
        # Hence why each branch of the graph begins with a zeroing of the grads.
        # Firstly, we zero the  z - z_bar. No gradients will flow back to z or z_bar from the tau loss function.
        with graph.gradient_override_map({'Identity': zero_the_grad}):
            tau_sq_alternate = tf.identity(tau_square_tile, name='Identity')
        # Then, we zero the  z - z_bar. No gradients will flow back to z or z_bar from the tau loss function.
        with graph.gradient_override_map({'Identity': zero_the_grad}):
            z_diff_sq_alternate = tf.identity(self.z_difference_squared, name='Identity')

        weighted_z_diff_sq, neg_dist, rbf = self._create_rbf_values(graph, self.z_difference_squared, tau_sq_alternate)
        weighted_variance, target_tau_diff, tau_loss = self._create_tau_loss(graph, z_diff_sq_alternate,
                                                                             tau_square_tile)

        # Calculate softmax and apply cross entropy loss function.
        sm = tf.nn.softmax(rbf)
        image_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=rbf)
        loss = tf.reduce_sum(image_loss) # + tau_loss

        return RbfTensors(z, z_bar, tau, sm, z_diff, self.z_difference_squared, weighted_z_diff_sq,
                          target_tau_diff, self.tau_quadratic_grad, self.z_grad, self.z_bar_grad, loss, rbf,
                          weighted_variance, neg_dist)

    def _tile_z(self, graph, z, batch_indices):
        """Tile z and define it's gradient modification.
        """
        z_grad_tag = "z_grad_"+self.network_id

        @tf.RegisterGradient(z_grad_tag)
        def _z_grad(unused_op, grad):
            """Modify the gradient of the cross-entropy loss function w.r.t z so training issues do not prohibit z
            from reach it's rbf cluster.

            Technically, this modifies the gradient of the loss w.r.t to the tiled version of z, this is z copied
            out from a m * d tensor to and m * d * K tensor. See the README for a the full details of how this
            function works and why it is designed this way (see the section on z).

            Args:
                unused_op: ignore

                grad: The gradient of the cross entropy loss function w.r.t to z_identity, with both the xe-softmax
                and rbf parts omitted.

            Returns: z_tile's modified gradient
            """
            grad = self._normalise(grad)
            _, _, K = grad.shape
            K = K.value
            y_hot_mask = self.xe_sm_grad - (1.0 - self.y_hot) * tf.maximum(self.xe_sm_grad, -0.1)
            y_hot_mask = tf.reshape(y_hot_mask, [-1, 1, K])
            new_grad = y_hot_mask * grad
            self.z_grad = new_grad
            return new_grad

        # If the batch indices argument was provided, use this to chop z down to just a batch (otherwise it will
        # already be in a batch).
        z_batch = z
        if batch_indices is not None:
            z_batch = tf.gather(z, batch_indices)

        # Tile the z's, so that they can be used for all m * d * k combinations of z - z_bar (broadcasting would take
        # care of this but the README specifies why this is done explicitly).
        z_re = tf.reshape(z_batch, [-1, self.d, 1])
        z_tile = tf.tile(z_re, [1, 1, self.num_class])
        with graph.gradient_override_map({'Identity': z_grad_tag}):
            z_tile = tf.identity(z_tile, name='Identity')
        return z, z_tile

    def _create_z_bar(self, graph):
        """Create and normalise the z_bars, then tile to a: m * d * K tensor from a d * K tensor. The normalisation is
        applied so they don't stray excessively. The gradient modifications to z_bar and z_bar_base are applied in
        this function.
        """
        z_bar_base_grad_tag = "z_bar_base_grad_"+self.network_id

        @tf.RegisterGradient(z_bar_base_grad_tag)
        def _z_bar_base_grad(unused_op, grad):
            """The gradient for z_bar_base.

                        Scale z_bar_base's gradient by z_bar_base's standard deviation. We do this because the natural gradient
                        is inversely scaled by z_bar_base's standard deviation, and we don't want that scaling so we're getting
                        rid of it.

                        Args:
                            unused_op: ignore

                            grad: The modified gradient of the xe loss function w.r.t to z_bar, multiplied by the natural
                            gradient of z_bar w.r.t to z_bar_base

                        Returns: The complete modified gradient of the xe loss function w.r.t z_bar_base
                        """
            return grad * (self.z_bar_sd + self.norm_epsilon)

        z_bar_base = tf.get_variable("z_bar_base", shape=[self.d, self.num_class], initializer=self.z_bar_init)
        with graph.gradient_override_map({'Identity': z_bar_base_grad_tag}):
            z_bar_base = tf.identity(z_bar_base, name='Identity')

        # Apply normalisation of z_bar_base to produce z_bar
        z_bar_mew = tf.reduce_mean(z_bar_base, axis=1)
        z_bar_mew = tf.reshape(z_bar_mew, shape=[self.d, 1])
        z_bar_diff = z_bar_base - z_bar_mew
        # Store of state in rbf object not ideal, but needed for gradient modification
        self.z_bar_sd = tf.reduce_mean(z_bar_diff ** 2.0, axis=1) ** 0.5
        self.z_bar_sd = tf.reshape(self.z_bar_sd, [self.d, 1])
        z_bar = 5.0 * z_bar_diff / (self.z_bar_sd + self.norm_epsilon)

        z_bar_grad_tag = "z_bar_grad_"+self.network_id

        @tf.RegisterGradient(z_bar_grad_tag)
        def _z_bar_grad(unused_op, grad):
            """Modify the gradient of the cross-entropy loss function w.r.t z_bar so training issues do not prohibit
            z_bar from reaching optima.

            Technically, this modifies the gradient of the loss w.r.t to the tiled version of z_bar, this is z_bar
            copied out from a d * K tensor to and m * d * K tensor. See the README for a the full details of how this
            function works and why it is designed this way (see the section on z_bar).

            Args:
                unused_op: ignore

                grad: The gradient of the cross entropy loss function w.r.t to z_identity, with both the xe-softmax
                and rbf parts omitted.

            Returns: z_bar_tile's modified gradient
            """
            _, d, K = grad.shape
            K = K.value
            grad = self._normalise(grad)
            xe_sm_grad_reshaped = tf.reshape(self.xe_sm_grad, [-1, 1, K])
            grad = xe_sm_grad_reshaped * grad
            self.z_bar_grad = grad / tf.cast(self.batch_size, tf.float32) ** 0.5
            return self.z_bar_grad * self.z_bar_lr_increase_factor

        # Tile the z_bar's, so that they can be used for all m * d * k combinations of z - z_bar (broadcasting would
        # take care of this but the README specifies why this is done explicitly).
        z_bar_tile = self._tile(z_bar)
        with graph.gradient_override_map({'Identity': z_bar_grad_tag}):
            z_bar_tile = tf.identity(z_bar_tile, name='Identity')

        return z_bar, z_bar_tile

    def _create_tau(self, graph):
        """Create tau, tile it, and define it's gradient modification.
        """
        tau_grad_tag = "tau_grad_"+self.network_id

        @tf.RegisterGradient(tau_grad_tag)
        def _tau_grad(unused_op, grad):
            """Modify the gradient of the cross-entropy loss function w.r.t tau so training issues do not prohibit
            tau from reaching optima.

            See the README for a the full details of how this function works (see the section on tau).

            Args:
                unused_op: ignore

                grad: thrown away, tau_quadratic_grad used instead (see README)

            Returns: tau's modified gradient
            """
            return tf.zeros(grad.shape) #TODO(Jack) HACK
            #return tf.sign(self.tau_quadratic_grad) * tf.abs(self.tau_quadratic_grad) ** 0.5 \
            #    / (tf.reduce_max(self.z_difference_squared, axis=0) + 10.0 ** -5.0) * self.tau_lr_increase_factor

        tau = tf.abs(tf.get_variable("tau", shape=[self.d, self.num_class],
                                     initializer=self.tau_init))
        with graph.gradient_override_map({'Identity': tau_grad_tag}):
            tau = tf.identity(tau, name='Identity')

        # Tile the taus's, so that they can be used for all m * d * k combinations of tau * (z - z_bar)
        return tau, self._tile(tau)

    def _create_rbf_values(self, graph, z_diff_sq, tau_sq_alternate):
        """Create the rbf values, i.e. the rbf probabilities scaled by the rbf constant.

        This function defines the part of the graph that takes the, m * d * K tensors (z - z_bar) ** 2.0 and
        tiled tau ** 2.0, and uses them to compute the scaled rbf probabilities. Two gradient modifications are applied,
        one to the gradient and another to the cross entropy softmax gradient.

        Args:
            z_diff_sq: The m * d * K tensor (z - z_bar) ** 2.0
            tau_sq_alternate: The m * d * K tensor tiled tau ** 2.0

        Returns: An m * K tensor of scaled rbf probabilities
        """
        weighted_z_diff_sq = tf.multiply(tau_sq_alternate, z_diff_sq)
        neg_dist = -tf.reduce_mean(weighted_z_diff_sq, axis=1)

        stub_rbf_grad_tag = "stub_rbf_grad_"+self.network_id

        @tf.RegisterGradient(stub_rbf_grad_tag)
        def _stub_rbf_grad(unused_op, grad):
            """Stub out the rbf gradient, by returning a tensor of ones in its place.

            At this point in the Backprop, the xe softmax gradient has been stubbed and saved, so effectively grad is
            just the derivative of rbf_c * exp w.r.t neg_dist, which is just rbf_c * exp. While the constant is benign
            the exponential is deadly and quickly causes the gradient to vanish. It's an unnecessary exponential scaling
            factor, so all this function does is remove it from the backward flow. See the README for more details.
            """
            return tf.ones(tf.shape(grad))

        with graph.gradient_override_map({'Identity': stub_rbf_grad_tag}):
            neg_dist = tf.identity(neg_dist, name='Identity')
        exp = tf.exp(neg_dist)
        rbf = self.rbf_c * exp

        stub_and_save_xe_sm_grad_tag = "stub_and_save_xe_sm_grad_"+self.network_id

        @tf.RegisterGradient(stub_and_save_xe_sm_grad_tag)
        def _stub_and_save(unused_op, grad):
            """Stub out the cross entropy softmax gradient (w.r.t rbf values), and save it so that it may be applied
            later, post normalisation.

            grad is the gradient of the loss with respect to the rbf values, i.e the cross entropy softmax signal
            y - a. The details of why this is applied to the rbf parameters later on is specified in the README.
            """
            self.xe_sm_grad = grad
            return tf.ones(tf.shape(grad))

        with graph.gradient_override_map({'Identity': stub_and_save_xe_sm_grad_tag}):
            rbf = tf.identity(rbf, name='Identity')
        return weighted_z_diff_sq, neg_dist, rbf

    def _create_tau_loss(self, graph, z_diff_sq_alternate, tau_square_tile):
        """Create tau's unique loss function, as specified in the README
        """
        # TODO(Jack) more comments / refactoring if you keep this
        fs_shape = tf.concat([[self.batch_size], [1, self.num_class]], axis=0)
        weighted_z_diff_sq_alternate = tau_square_tile * z_diff_sq_alternate
        filtered_sum = tf.reshape(self.y_hot, fs_shape) * weighted_z_diff_sq_alternate
        class_wise_batch_size = tf.reduce_sum(self.y_hot, axis=0)
        is_greater_than_zero = tf.greater(class_wise_batch_size, 0.01)
        ones = tf.ones([self.num_class])
        safe_class_wise_batch_size = tf.where(is_greater_than_zero, class_wise_batch_size, ones)
        safe_class_wise_batch_size = tf.reshape(safe_class_wise_batch_size, [1, self.num_class])
        weighted_variance = tf.reduce_sum(filtered_sum, axis=0) / safe_class_wise_batch_size

        stub_and_save_tau_quadratic_grad_tag = "stub_and_save_tau_quadratic_grad_"+self.network_id

        @tf.RegisterGradient(stub_and_save_tau_quadratic_grad_tag)
        def _stub_and_save_tau_quadratic_grad(unused_op, grad):
            self.tau_quadratic_grad = grad
            return tf.ones(tf.shape(grad))

        with graph.gradient_override_map({'Identity': stub_and_save_tau_quadratic_grad_tag}):
            weighted_variance = tf.identity(weighted_variance, name='Identity')
        target_tau_diff = (self.target_precision - weighted_variance) ** 2.0
        tau_loss = tf.reduce_sum(target_tau_diff)
        return weighted_variance, target_tau_diff, tau_loss

    def _tile(self, rbf_param):
        tile_shape = tf.concat([[self.batch_size], [1, 1]], axis=0)
        reshaped = tf.reshape(rbf_param, [1, self.d, -1])
        return tf.tile(reshaped, tile_shape)

    def _normalise(self, grad):
        # Normalise a gradient using a 1-norm.
        _, _, K = grad.shape
        K = K.value
        grad_mag = tf.reduce_sum(tf.abs(grad), axis=1)  # tf.reduce_sum(grad ** 2.0, axis=1) ** 0.5
        normalised = grad / (tf.reshape(grad_mag, [-1, 1, K]) + self.norm_epsilon)
        return normalised


class RbfTensors(object):
    """This class encapsulates all the tensors in the rbf part of the graph that are needed by other components."""

    def __init__(self, z, z_bar, tau, a, z_diff, z_diff_sq, weighted_z_diff_sq, target_tau_diff, tau_quadratic_grad,
               z_grad, z_bar_grad, loss, rbf, weighted_variance, neg_dist):
        self.z = z
        self.z_bar = z_bar
        self.tau = tau
        self.a = a
        self.z_diff = z_diff
        self.z_diff_sq = z_diff_sq
        self.wzds = weighted_z_diff_sq
        self.target_tau_diff = target_tau_diff
        self.tau_quadratic_grad = tau_quadratic_grad
        self.z_grad = z_grad
        self.z_bar_grad = z_bar_grad
        self.loss = loss
        self.rbf = rbf
        self.weighted_variance = weighted_variance
        self.neg_dist = neg_dist

    def core_ops(self):
        return [self.loss, self.z, self.z_bar, self.tau, self.a, self.rbf, self.weighted_variance, self.z_diff,
                self.neg_dist, self.wzds]
