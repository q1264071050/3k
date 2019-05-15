import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helper
from tensorflow.contrib import learn
from text_cnn import TextCNN

os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # 指定一个GPU
# parameters

# Data loadding  params
tf.flags.DEFINE_float(name='dev_sample_percentage', default=0.1,
                      help='Percentage of the training data to use for validation')
tf.flags.DEFINE_string(name='positive_data_file', default='./data/pos_data.txt',
                       help='positive data')
tf.flags.DEFINE_string(name='negative_data_file', default='./data/neg_data.txt',
                       help='negative data')
tf.flags.DEFINE_integer(name="num_labels",  default=2, help="Number of labels for data. (default: 2)")
# Model hyperparams
tf.flags.DEFINE_integer(name='embedding_dim', default=64, help='dimensionality of word')
#tf.flags.DEFINE_integer(name='padding_size', default=100, help='padding seize of eatch sample')
tf.flags.DEFINE_string(name='filter_size', default='3,4,5', help='filter size ')
tf.flags.DEFINE_integer(name='num_filters', default=128, help='deep of filters')
tf.flags.DEFINE_float(name='dropout_keep_prob', default=0.5, help='Drop out')
tf.flags.DEFINE_float(name='L2_reg_lambda', default=0.0, help='L2')

# Training params
tf.flags.DEFINE_integer(name='batch_size', default=64, help='batch size')
tf.flags.DEFINE_float(name='learning_rate', default=0.1, help='learning rate')
tf.flags.DEFINE_integer(name="num_epochs",  default=200, help="Number of training epochs (default: 200)")
tf.flags.DEFINE_integer(name="evaluate_every", default=100,
                        help="Evalue model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer(name="checkpoint_every", default=100,  help="Save model after this many steps (defult: 100)")
tf.flags.DEFINE_integer(name="num_checkpoints", default=5,  help="Number of checkpoints to store (default: 5)")

# Misc parameters
tf.flags.DEFINE_boolean(name='allow_soft_placement', default='True',
                        help='allow_soft_placement')  # 找不到指定设备时，是否自动分配
tf.flags.DEFINE_boolean(name='log_device_placement', default='False',
                        help='log_device_placement ')  # 是否打印配置日志

# Parse parameters from commands
FLAGS = tf.flags.FLAGS
FLAGS.flag_values_dict()  # 解析参数成字典
#FLAGS._parse_flags()

print('\n----------------Parameters--------------')  # 在网络训练之前，先打印出来看看
for attr, value in sorted(FLAGS.flag_values_dict().items()):
    print('{}={}'.format(attr.upper(), value))

# Prepare output directory for models and summaries
# =======================================================

timestamp = str(int(time.time()))
out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
print("Writing to {}\n".format(out_dir))
if not os.path.exists(out_dir):
    os.makedirs(out_dir)


# Load data and cut
x_text, y = data_helper.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)

# Get embedding vector
sentences, max_document_length = data_helper.padding_sentence(x_text, '<PADDING>')
x = np.array(data_helper.embedding_sentences(sentences,
                                             embedding_size=FLAGS.embedding_dim,
                                             file_to_save=os.path.join(out_dir, 'trained_word2vec.model')))
print("x.shape = {}".format(x.shape))
print("y.shape = {}".format(y.shape))

# Save params
training_params_file = os.path.join(out_dir, 'training_params.pickle')
params = {'num_labels': FLAGS.num_labels, 'max_document_length': max_document_length}
data_helper.saveDict(params, training_params_file)

# Shuffle data randomly
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]


# Split train/test set
# TODO: This is very crude, should use cross-validation
dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,
                                  log_device_placement=FLAGS.log_device_placement)
    session_conf.gpu_options.per_process_gpu_memory_fraction = 0.6
    session_conf.gpu_options.allow_growth = True
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(sequence_length=x_train.shape[1],
                      num_classes=y_train.shape[1],
                      embedding_size=FLAGS.embedding_dim,
                      filter_sizes=list(map(int, FLAGS.filter_size.split(','))),
                      num_filters=FLAGS.num_filters,
                      l2_reg_lambda=FLAGS.L2_reg_lambda
                      )
        global_step = tf.Variable(0, trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        with tf.device('/gpu:1'):
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
    sess.run(tf.global_variables_initializer())
    last = datetime.datetime.now()

    grad_summaries = []
    for g, v in grads_and_vars:
        if g is not None:
            grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
            sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
            grad_summaries.append(grad_hist_summary)
            grad_summaries.append(sparsity_summary)
    grad_summaries_merged = tf.summary.merge(grad_summaries)

    # Output directory for models and summaries
    print("Writing to {}\n".format(out_dir))

    # Summaries for loss and accuracy
    loss_summary = tf.summary.scalar("loss", cnn.loss)
    acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

    # Train Summaries
    train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
    train_summary_dir = os.path.join(out_dir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

    # Dev summaries
    dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
    dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
    dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

    # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

    # Initialize all variables
    sess.run(tf.global_variables_initializer())


    def train_step(x_batch, y_batch):
        """
        A single training step
        """
        feed_dict = {
            cnn.input_x: x_batch,
            cnn.input_y: y_batch,
            cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
        }
        _, step, summaries, loss, accuracy = sess.run(
            [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
            feed_dict)
        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
        train_summary_writer.add_summary(summaries, step)


    def dev_step(x_batch, y_batch, writer=None):
        """
        Evaluates model on a dev set
        """
        feed_dict = {
            cnn.input_x: x_batch,
            cnn.input_y: y_batch,
            cnn.dropout_keep_prob: 1.0
        }
        step, summaries, loss, accuracy = sess.run(
            [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
            feed_dict)
        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
        if writer:
            writer.add_summary(summaries, step)


    # Generate batches
    batches = data_helper.batch_iter(
        list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)

    # Training loop. For each batch...
    for batch in batches:
        x_batch, y_batch = zip(*batch)
        train_step(x_batch, y_batch)
        current_step = tf.train.global_step(sess, global_step)
        if current_step % FLAGS.evaluate_every == 0:
            print("\nEvaluation:")
            dev_step(x_dev, y_dev, writer=dev_summary_writer)
            print("")
        if current_step % FLAGS.checkpoint_every == 0:
            path = saver.save(sess, checkpoint_prefix, global_step=current_step)
            print("Saved model checkpoint to {}\n".format(path))


