import tensorflow as tf

def get_MLP(s1, s2, num_features, num_actions, num_neurons, num_hidden_layers):
    # Instructions to update the target network
    update_target = []
    # First layer
    layer, layer_t = _add_layer(s1, s2, num_features, num_neurons, True, update_target, 0)
    # Hidden layers
    for i in range(num_hidden_layers):
        layer, layer_t = _add_layer(layer, layer_t, num_neurons, num_neurons, True, update_target, i+1)
    # Output Layer
    q_values, q_target = _add_layer(layer, layer_t, num_neurons, num_actions, False, update_target, num_hidden_layers+1)
    return q_values, q_target, update_target

def _add_layer(s, s_t, num_input, num_output, use_relu, update, id):
    layer, W, b = _add_dense_layer(s, num_input, num_output, True, id)
    layer_t, W_t, b_t = _add_dense_layer(s_t, num_input, num_output, False, id)
    update.extend([tf.assign(W_t,W), tf.assign(b_t,b)])
    if use_relu:
        layer = tf.nn.relu(layer)
        layer_t = tf.nn.relu(layer_t)
    return layer, layer_t

def _add_dense_layer(s, num_input, num_output, is_trainable, id):
    W = tf.Variable(tf.truncated_normal([num_input, num_output], stddev=0.1, dtype=tf.float64), trainable = is_trainable, name="W"+str(id))
    b = tf.Variable(tf.constant(0.1, shape=[num_output], dtype=tf.float64), trainable = is_trainable, name="b"+str(id))
    return tf.matmul(s, W) + b, W, b