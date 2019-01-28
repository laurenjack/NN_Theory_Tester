import numpy as np


def assert_sess_run_called_with(session_mock, expected_tensors, expected_feed_dicts):
    """Assert that session.run was called with specific arguments (must specify for every call).

    Validating methods that are called with numpy arrays is difficult due to numpy's error throwing when == is called.
    We can unpack the arguments from the mock object manually and use numpy.array_equal, this method handles that
    complexity. It's specifically designed for calls to tensorflows Session.run, as the feed_dict argument involves
    numpy arrays within a dictionary.

    Args:
        session_mock: The Mock object used for the tensorflow Session object (in production code session is the
        tensorflow session).
        tensors: A list of: tensors, or lists of tensors (probably mocks) where each element is the next tensor argument
        that we expect to be passed to Session.run, in call order.
        feed_dict: The feed_dicts we expect to be passed to Session.run, in call order.

    Raises:
        An AssertionError if Session.Run() was not called as we expected
    """
    call_args_list = session_mock.run.call_args_list
    if len(call_args_list) != len(expected_tensors) != len(expected_feed_dicts):
        raise AssertionError('The number of actual calls to session.run, expected tensor arguments and expected'
                             'arguments must all be equal.')

    for call_args, tensor_arg, feed_dict in zip(call_args_list, expected_tensors, expected_feed_dicts):
        _assert_session_dot_run_for_single_call(call_args, tensor_arg, feed_dict)


def _assert_session_dot_run_for_single_call(actual_call_args, expected_tensor_arg, expected_feed_dict):
    ordered_args, keyword_args = actual_call_args
    if len(ordered_args) != 1 or 'feed_dict' not in keyword_args.keys():
        raise AssertionError('Need:\n'
                             '1) A mock tensor, list, or tuple of mock tensors as the first call to session.run\n'
                             '2) A keyword argument for feed_dict')

    actual_tensor_arg = ordered_args[0]
    if expected_tensor_arg != actual_tensor_arg:
        raise AssertionError('session.run was called with the ordered argument:\n{actual} but we expected:\n{expected}'
                             .format(actual=actual_tensor_arg, expected=expected_tensor_arg))

    feed_dict_arg = keyword_args['feed_dict']
    if not _are_dictionaries_equal(feed_dict_arg, expected_feed_dict):
        raise AssertionError('session.run was called with the feed_dict:\n{actual} but we expected:\n{expected}'
                             .format(actual=feed_dict_arg, expected=expected_feed_dict))


def _are_dictionaries_equal(d1, d2):
    # The set of keys must be the same
    if d1.keys() != d2.keys():
        return False
    # If the keys are the same, this loop will infer if the values are the same
    for k1, v1 in d1.iteritems():
        v2 = d2[k1]
        # If we have a numpy array, use the array equals method for comparision
        if isinstance(v2, np.ndarray):
            if not np.array_equal(v1, v2):
                return False
        # Otherwise the inequality operator should be OK
        else:
            if v1 != v2:
                return False
    return True
