import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import nn_ops, math_ops, array_ops, init_ops
from tensorflow.python.util import nest
from operator import mul

_BIAS_VARIABLE_NAME = "biases"
_WEIGHTS_VARIABLE_NAME = "weights"


def _linear(args, output_size, bias, bias_start=0.0):
  """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    bias: boolean, whether to add a bias term or not.
    bias_start: starting value to initialize the bias; 0 by default.
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

  # Now the computation.
  scope = vs.get_variable_scope()
  with vs.variable_scope(scope) as outer_scope:
    weights = vs.get_variable(
        _WEIGHTS_VARIABLE_NAME, [total_arg_size, output_size], dtype=dtype)
    if len(args) == 1:
      res = math_ops.matmul(args[0], weights)
    else:
      res = math_ops.matmul(array_ops.concat(args, 1), weights)
    if not bias:
      return res
    with vs.variable_scope(outer_scope) as inner_scope:
      inner_scope.set_partitioner(None)
      biases = vs.get_variable(
          _BIAS_VARIABLE_NAME, [output_size],
          dtype=dtype,
          initializer=init_ops.constant_initializer(bias_start, dtype=dtype))
    return nn_ops.bias_add(res, biases)


def get_logits(args, output_size, bias, input_keep_prob=1.0, is_train=None):
    def flatten(tensor, keep):
        fixed_shape = tensor.get_shape().as_list()
        start = len(fixed_shape) - keep
        left = reduce(mul, [fixed_shape[i] or tf.shape(tensor)[i] for i in range(start)])
        out_shape = [left] + [fixed_shape[i] or tf.shape(tensor)[i] for i in range(start, len(fixed_shape))]
        flat = tf.reshape(tensor, out_shape)
        return flat

    def reconstruct(tensor, ref, keep):
        ref_shape = ref.get_shape().as_list()
        tensor_shape = tensor.get_shape().as_list()
        ref_stop = len(ref_shape) - keep
        tensor_start = len(tensor_shape) - keep
        pre_shape = [ref_shape[i] or tf.shape(ref)[i] for i in range(ref_stop)]
        keep_shape = [tensor_shape[i] or tf.shape(tensor)[i] for i in range(tensor_start, len(tensor_shape))]
        target_shape = pre_shape + keep_shape
        out = tf.reshape(tensor, target_shape)
        return out

    flat_args = [flatten(arg, 1) for arg in args]
    flat_args = [tf.cond(is_train, lambda: tf.nn.dropout(arg, input_keep_prob), lambda: arg) for arg in flat_args]

    with tf.variable_scope("Linear_Logits"):
        flat_logits = _linear(args=flat_args, output_size=output_size, bias=bias)

    logits = reconstruct(flat_logits, args[0], 1)
    logits = tf.squeeze(logits, [len(args[0].get_shape().as_list())-1])
    return logits
