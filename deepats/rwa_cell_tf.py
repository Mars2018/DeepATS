import collections
import tensorflow as tf

# from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell
from tensorflow.contrib.rnn import RNNCell
# from tensorflow.nn.rnn_cell import RNNCell

from tensorflow.contrib.rnn.python.ops import core_rnn_cell_impl
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.ops import variable_scope as vs

# from utils import linear
from tensorflow.python.util import nest
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops

_checked_scope = core_rnn_cell_impl._checked_scope
RWACellTuple = collections.namedtuple("RWACellTuple", ("h", "n", "d", "a_max"))

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"

def linear(args,
           output_size,
           bias,
           bias_initializer=None,
           kernel_initializer=None,
           kernel_regularizer=None,
           bias_regularizer=None,
           normalize=False):
  """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    bias: boolean, whether to add a bias term or not.
    bias_initializer: starting value to initialize the bias
      (default is all zeros).
    kernel_initializer: starting value to initialize the weight.
    kernel_regularizer: kernel regularizer
    bias_regularizer: bias regularizer
  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """
  if args is None or (nest.is_sequence(args) and not args):
    raise ValueError("`args` must be specified")
  if not nest.is_sequence(args):
    args = [args]

  # Calculate the total size of arguments on dimension 1.
  total_arg_size = 0
  shapes = [a.get_shape() for a in args]
  for shape in shapes:
    if shape.ndims != 2:
      raise ValueError("linear is expecting 2D arguments: %s" % shapes)
    if shape[1].value is None:
      raise ValueError("linear expects shape[1] to be provided for shape %s, "
                       "but saw %s" % (shape, shape[1]))
    else:
      total_arg_size += shape[1].value

  dtype = [a.dtype for a in args][0]

class RWACell(RNNCell):
  """Recurrent Weighted Average (cf. http://arxiv.org/abs/1703.01253)."""

  def __init__(self, num_units, input_size=None, activation=tanh, normalize=False, reuse=None):
    if input_size is not None:
      tf.logging.warn("%s: The input_size parameter is deprecated.", self)
    self._num_units = num_units
    self._activation = activation
    self._reuse = reuse
    self._normalize = normalize

  @property
  def state_size(self):
    return RWACellTuple(self._num_units, self._num_units, self._num_units, self._num_units)

  def zero_state(self, batch_size, dtype):
    h, n, d, _ = super(RWACell, self).zero_state(batch_size, dtype)
    a_max = tf.fill([batch_size, self._num_units], -1E38) # Start off with lowest number possible
    return RWACellTuple(h, n, d, a_max)

  @property
  def output_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope=None):
    with _checked_scope(self, scope or "rwa_cell", reuse=self._reuse):
      h, n, d, a_max = state

      with vs.variable_scope("u"):
        u = linear(inputs, self._num_units, True, normalize=self._normalize)

      with vs.variable_scope("g"):
        g = linear([inputs, h], self._num_units, True, normalize=self._normalize)

      with vs.variable_scope("a"): # The bias term when factored out of the numerator and denominator cancels and is unnecessary
        a = linear([inputs, h], self._num_units, False, normalize=self._normalize)

      z = tf.multiply(u, tanh(g))

      a_newmax = tf.maximum(a_max, a)
      exp_diff = tf.exp(a_max - a_newmax)
      exp_scaled = tf.exp(a - a_newmax)

      n = tf.multiply(n, exp_diff) + tf.multiply(z, exp_scaled)  # Numerically stable update of numerator
      d = tf.multiply(d, exp_diff) + exp_scaled  # Numerically stable update of denominator
      h_new = self._activation(tf.div(n, d))

      new_state = RWACellTuple(h_new, n, d, a_newmax)

    return h_new, new_state