#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helper
from text_cnn import TextCNN
import csv
start =time.clock()
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# Parameters参数
# ==================================================

# Data Parameters数据参数
tf.flags.DEFINE_string("input_text_file", "./data/test_data.txt", "Test text data source to evaluate.")
tf.flags.DEFINE_string("input_label_file", "", "Label file for test text data source.")

# Eval Parameters预测参数
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "./runs/1557136083/checkpoints", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", True, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS.flag_values_dict()
print("\nParameters:")
for attr, value in sorted(FLAGS.flag_values_dict().items()):
    print("{}={}".format(attr.upper(), value))
print("")

# validate生效
# ==================================================

# validate checkout point file
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
if checkpoint_file is None:
    print("Cannot find a valid checkpoint file!")
    exit(0)
print("Using checkpoint file : {}".format(checkpoint_file))

# validate word2vec model file验证w2v模型文件

trained_word2vec_model_file = ("./runs/1557136083/trained_word2vec.model")
if not os.path.exists(trained_word2vec_model_file):
    print("Word2vec model file \'{}\' doesn't exist!".format(trained_word2vec_model_file))
print("Using word2vec model file : {}".format(trained_word2vec_model_file))

# validate training params file

training_params_file = ("./runs/1557136083/training_params.pickle")
if not os.path.exists(training_params_file):
    print("Training params file \'{}\' is missing!".format(training_params_file))
print("Using training params file : {}".format(training_params_file))

# Load params
params = data_helper.loadDict(training_params_file)
num_labels = int(params['num_labels'])
max_document_length = int(params['max_document_length'])

# Load data
if FLAGS.eval_train:
    x_raw, y_test = data_helper.load_test_data(FLAGS.input_text_file), None
else:
    x_raw = ["a masterpiece four years in the making", "everything is off."]
    y_test = [1, 0]

# Get Embedding vector x_test
sentences, max_document_length = data_helper.padding_sentence(x_raw, '<PADDING>', padding_sentence_length = max_document_length)
x_test = np.array(data_helper.embedding_sentences(sentences, file_to_load = trained_word2vec_model_file))
print("x_test.shape = {}".format(x_test.shape))


# Evaluation
# ==================================================
print("\nEvaluating...\n")
# checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("out-fc/prediction").outputs[0]

        # Generate batches for one epoch
        batches = data_helper.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []

        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])

# Print accuracy if y_test is defined
if y_test is not None:
    correct_predictions = float(sum(all_predictions == y_test))
    print("Total number of test examples: {}".format(len(y_test)))
    print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))

# Save the evaluation to a csv
predictions_readable = np.column_stack((np.array([''.join(text).split('<PADDING>')[0] for text in x_raw]), all_predictions))
out_path = ( "./runs/1557136083/prediction.csv")
print("Saving evaluation to {0}".format(out_path))
with open(out_path, 'w', newline='') as f:
    csv.writer(f).writerows(predictions_readable)
end=time.clock()
print('Running time: %s Seconds'%(end-start))