"""TensorFlow benchmark library for NLP"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os
from models.nlp import bert_model
from models.nlp import optimization
from models.nlp import tokenization
from models.nlp.transformer.model import model_params
from models.nlp.transformer.model import schedule
from models.nlp.transformer.utils import metrics
from models.nlp.transformer.utils import dataset
from models.nlp.transformer.model import transformer
import tensorflow as tf

###########################################################
# Functions and classes for BERT benchmark
###########################################################
class InputExample(object):
  """A training/test example for sequence classification."""

  def __init__(self, guid, text_a, text_b=None, label=None):
    """Constructs a InputExample.

    Args:
      guid: id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only specifies this sequence.
      text_b: string. [Optional] The untokenized text of the second sequence.
        Only be specified for sequence pair tasks.
      label: string. [Optional] The label of the example.
    """
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label

class PaddingInputExample(object):
  """Fake example so the num input examples is a multiple of the batch size."""

class InputFeatures(object):
  """A set of features of data."""

  def __init__(self,
               input_ids,
               input_mask,
               segment_ids, 
               label_id,
               is_real_example=True):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_id = label_id
    self.is_real_example = is_real_example

class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

  def get_train_examples(self, data_dir):
    """Gets a collection of 'InputExample's for the train set."""
    raise NotImplementedError()

  def get_dev_examples(self, data_dir):
    """Gets a collection of 'InputExample's for the dev set."""
    raise NotImplementedError()

  def get_test_examples(self, data_dir):
    """Gets a collection of 'InputExample's for predication."""
    raise NotImplementedError()
  
  def get_labels(self):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()

  @classmethod
  def _read_tsv(cls, input_file, quotechar=None):
    """Reads a tab seperated value file."""
    with tf.gfile.Open(input_file, "r") as f:
      reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
      lines = []
      for line in reader:
        lines.append(line)
      return lines

class XnliProcessor(DataProcessor):
  """Processor for the XNLI data set."""

  def __init__(self):
    self.language = "zh"

  def get_train_examples(self, data_dir):
    """See base class."""
    lines = self._read_tsv(os.path.join(data_dir, 'multinli',
                           'multinli.train.%s.tsv' % self.language))
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "train-%d" % (i)
      text_a = tokenization.convert_to_unicode(line[0])
      text_b = tokenization.convert_to_unicode(line[1])
      label = tokenization.convert_to_unicode(line[2])
      if label == tokenization.convert_to_unicode('contradictory'):
        label = tokenization.convert_to_unicode('contradiction')
      examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b,
                      label=label))
    return examples
      
  def get_dev_examples(self, data_dir):
    """See base class."""
    lines = self._read_tsv(os.path.join(data_dir, 'xnli.dev.tsv'))
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "dev-%d" % (i)
      language = tokenization.convert_to_unicode(line[0])
      if language != tokenization.convert_to_unicode(self.language):
        continue
      text_a = tokenization.convert_to_unicode(line[6])
      text_b = tokenization.convert_to_unicode(line[7])
      label = tokenization.convert_to_unicode(line[1])
      examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b,
                      label=label))
    return examples
  
  def get_labels(self):
    """See base class."""
    return ["contradiction", "entailment", "neutral"]

class MnliProcessor(DataProcessor):
  """Processor for the MultiNLI data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")), 
            "dev_matched")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test_matched.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return ["contradiction", "entailment", "neutral"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "%s-%s" % (set_type, tokenization.convert_to_unicode(line[0]))
      text_a = tokenization.convert_to_unicode(line[8])
      text_b = tokenization.convert_to_unicode(line[9])
      if set_type == "test":
        label = "contradiction"
      else:
        label = tokenization.convert_to_unicode(line[-1])
      examples.append(guid=guid, text_a=text_a, text_b=text_b, label=label)
    return examples

class MrpcProcessor(DataProcessor):
  """Processor for the MRPC data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "%s-%s" % (set_type, i)
      text_a = tokenization.convert_to_unicode(line[3])
      text_b = tokenization.convert_to_unicode(line[4])
      if set_type == "test":
        label = "0"
      else:
        label = tokenization.convert_to_unicode(line[0])
      examples.append(
              InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples

class ColaProcessor(DataProcessor):
  """Processor for the CoLA data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      # Only the test set has a header
      if set_type == "test" and i == 0:
        continue
      guid = "%s-%s" % (set_type, i)
      if set_type == "test":
        text_a = tokenization.convert_to_unicode(line[1])
        label = "0"
      else:
        text_a = tokenization.convert_to_unicode(line[3])
        label = tokenization.convert_to_unicode(line[1])
      examples.append(guid=guid, text_a=text_a, text_b=None, label=label)
    return examples

def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
  """Converts a single 'InputExample' into a single 'InputFeature'."""
  pass
  if isinstance(example, PaddingInputExample):
    return InputFeatures(
            input_ids=[0] * max_seq_length,
            input_mask=[0] * max_seq_length,
            segment_ids=[0] * max_seq_length,
            label_id=0,
            is_real_example=False
            )
  label_map = {}
  for (i, label) in enumerate(label_list):
    label_map[label] = i

  tokens_a = tokenizer.tokenize(example.text_a)
  tokens_b = None
  if example.text_b:
    tokens_b = tokenizer.tokenize(example.text_b)
  if tokens_b:
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
  else:
    if len(tokens_a) > max_seq_length - 2:
      tokens_a = tokens_a[0:(max_seq_length - 2)]

  tokens = []
  segment_ids = []
  tokens.append("[CLS]")
  segment_ids.append(0)
  for token in tokens_a:
    tokens.append(token)
    segment_ids.append(0)

  tokens.append("[SEP]")
  segment_ids.append(0)

  if tokens_b:
    for token in tokens_b:
      tokens.append(token)
      segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

  input_ids = tokenizer.convert_tokens_to_ids(tokens)

  input_mask = [1] * len(input_ids)

  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length

  label_id = label_map[example.label]
  if ex_index < 5:
    tf.logging.info("*** Example ***")
    tf.logging.info("guid: %s" % (example.guid))
    tf.logging.info("tokens: %s" % " ".join(
        [tokenization.printable_text(x) for x in tokens]))
    tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    tf.logging.info("label: %s (id = %d)" % (example.label, label_id))

  feature = InputFeatures(
          input_ids=input_ids,
          input_mask=input_mask,
          segment_ids=segment_ids,
          label_id=label_id,
          is_real_example=True)

  return feature

def file_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, output_file):
  "Convert a set of 'InputExample's to a TFRecord file."""

  writer = tf.python_io.TFRecordWriter(output_file)

  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    feature = convert_single_example(ex_index, example, label_list, 
                                     max_seq_length, tokenizer)

    def create_int_feature(values):
      f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
      return f

    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["input_mask"] = create_int_feature(feature.input_mask)
    features["segment_ids"] = create_int_feature(feature.segment_ids)
    features["label_ids"] = create_int_feature([feature.label_id])
    features["is_real_example"] = create_int_feature(
            [int(feature.is_real_example)])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())
  writer.close()

def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
  """Creates an 'input_fn' closure to be passed to Estimator."""

  name_to_features = {
          "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
          "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
          "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
          "label_ids": tf.FixedLenFeature([], tf.int64),
          "is_real_example": tf.FixedLenFeature([], tf.int64),
        }

  def _decode_record(record, name_to_features):
    """Decodes a record to a Tensorflow example."""
    example = tf.parse_single_example(record, name_to_features)

    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      example[name] = t

    return example

  def input_fn(params):
    """The actual input functions."""
    batch_size = params["batch_size"]

    d = tf.data.TFRecordDataset(input_file)
    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.apply(
          tf.contrib.data.map_and_batch(
              lambda record: _decode_record(record, name_to_features),
              batch_size=batch_size,
              drop_remainder=drop_remainder))
    return d
  return input_fn
  

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""
  
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()

def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
        labels, num_labels):
  """Creates a classfication mode."""
  model = bert_model.BertModel(
          config=bert_config,
          is_training=is_training,
          input_ids=input_ids,
          input_mask=input_mask,
          token_type_ids=segment_ids)

  # In the demo, we are doing a simple classification task on the entire
  # segment.
  #
  # If you want to use the token-level output, use model.get_sequence_output()
  # instead.
  output_layer = model.get_pooled_output()

  hidden_size = output_layer.shape[-1].value

  output_weights = tf.get_variable(
          "output_weights", [num_labels, hidden_size],
          initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
          "output_bias", [num_labels], initializer=tf.zeros_initializer())

  with tf.variable_scope("loss"):
    if is_training:
      # I.e., 0.1 dropout
      output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    probabilities = tf.nn.softmax(logits, axis=-1)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)

    return (loss, per_example_loss, logits, probabilities)

def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
        num_train_steps, num_warmup_steps):
  """Returns 'model_fn' closure for Estimator."""
  def model_fn(features, labels, mode, params):
    """The 'model_fn' for Estimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label_ids"]
    is_real_example = None
    if "is_real_example" in features:
      is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
    else:
      is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    (total_loss, per_example_loss, logits, probabilities) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids,
            label_ids, num_labels)

    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = bert_model.get_assignment_map_from_checkpoint(tvars, assignment_map)
      tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("*** Trainable Variables ***")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
              init_string)

    ouput_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:
      train_op = optimization.create_optimizer(
              total_loss, learning_rate, num_train_steps, num_warmup_steps,
              False)
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
              mode=mode,
              loss=total_loss,
              train_op=train_op,
              scaffold_fn=scaffold_fn)
    elif mode == tf.estimator.ModeKeys.EVAL:
      def metric_fn(per_example_loss, label_ids, logits, is_real_example):
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        accuracy = tf.metrics.accuracy(
            labels=label_ids, predictions=predictions, weights=is_real_example)
        loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
        return {
            "eval_accuracy": accuracy,
            "eval_loss": loss,
        }

      eval_metrics = (metric_fn,
                      [per_example_loss, label_ids, logits, is_real_example])
      output_spec = tf.contrib.tpu.TPUEstimator(
              mode=mode,
              loss=total_loss,
              eval_metrics=eval_metrics,
              scaffold_fn=scaffold_fn)
    else:
      output_spec = tf.contrib.tpu.TPUEstimator(
              mode=mode,
              predictions={"probabilities": probabilities},
              scaffold_fn=scaffold_fn)
    return output_spec

  return model_fn

def input_fn_builder(features, seq_length, is_training, drop_remainder):
  """Creates an 'input_fn' closure to be passed to Estimator."""

  all_input_ids = []
  all_input_mask = []
  all_segment_ids = []
  all_label_ids = []

  for feature in features:
    all_input_ids.append(feature.input_ids)
    all_input_mask.append(feature.input_mask)
    all_segment_ids.append(feature.segment_ids)
    all_label_ids.append(feature.label_id)

  def input_fn(params):
    """The actual input function."""
    batch_size = params.batch_size

    num_examples = len(features)

    # This is for demo purposes and NOT scale to large data sets. We do
    # not use Dataset.from_generator() because that uses tf.py_func which is
    # not TPU compatible. The right way to load data is with TFRecordReader.
    d = tf.data.Dataset.from_tensor_slices({
        "input_ids":
          tf.constant(
              all_input_ids, shape=[num_examples, seq_length],
              dtype=tf.int32),
        "input_mask":
          tf.constant(
              all_input_mask, shape=[num_examples, seq_length],
              dtype=tf.int32),
        "segment_ids":
          tf.constant(
              all_segment_ids, shape=[num_examples, seq_length],
              dtype=tf.int32),
        "label_ids":
          tf.constant(
              all_label_ids, shape=[num_examples], dtype=tf.int32),
        })

    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
    return d

  return input_fn

def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer):
  """Convert a set of 'InputExample's to a list of 'InputFeatures'."""

  features = []
  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    feature = convert_single_example(ex_index, example, label_list,
                                     max_seq_length, tokenizer)
    features.append(feature)
  return features

###########################################################
# Functions and classes for transformer benchmark
###########################################################
PARAMS_MAP = {
        'tiny': model_params.TINY_PARAMS,
        'base': model_params.BASE_PARAMS,
        'big': model_params.BIG_PARAMS,
}

DEFAULT_TRAIN_EPOCHS = 10
INF = int(1e9)
BLEU_DIR = "bleu"

def model_fn(features, labels, mode, params):
  """Define how to train, evaluate and predict from the transformer mode."""
  with tf.variable_scope("model"):
    inputs, targets = features, labels

    # Create model and get output logits.
    model=transformer.Transformer(params, mode == tf.estimator.ModeKeys.TRAIN)

    logits = model(inputs, targets)

    # When in prediction mode, the labels/targets is None. The model output
    # is the prediction.
    if mode == tf.estimator.ModeKeys.PREDICT:
      return tf.estimator.EstimatorSpec(
              tf.estimator.ModeKeys.PREDICT,
              predictions=logits,
              export_outputs={
                  'translate': tf.estimator.export.PredictOutput(logits)
             })

    logits.set_shape(targets.shape.as_list() + logits.shape.as_list()[2:])

    # Calculate model loss.
    # xentropy contains the cross entropy loss of every nonpadding token in
    # the targets.
    xentropy, weights = metrics.padded_cross_entropy_loss(
            logits, targets, params["label_smoothing"], params["vocab_size"])
    loss = tf.reduce_sum(xentropy) / tf.reduce_sum(weights)

    # Save loss as named tensor that will be logged with the logging hook.
    tf.identity(loss, 'cross_entropy')

    if mode == tf.estimator.ModeKeys.EVAL:
      return tf.estimator.EstimatorSpec(
              mode=mode, loss=loss, predictions={'predictions': logits},
              eval_metric_ops=metrics.get_eval_metrics(logits, labels, params))
    else:
      train_op, metric_dict = get_train_op_and_metrics(loss, params)

      # Epochs can be quite long. This gives some intermediate information
      # in TensorBoard.
      metric_dict["minibatch_loss"] = loss
      record_scalars(metric_dict)
      return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

def record_scalars(metric_dict):
  for key, value in metric_dict.items():
    tf.contrib.summary.scalar(name=key, value=value)

def get_learning_rate(learning_rate, hidden_size, learning_rate_warmup_steps):
  """Calculates learning rate with linear warmup and rsqrt decay."""
  with tf.name_scope("learning_rate"):
    warmup_steps = tf.to_float(learning_rate_warmup_steps)
    step = tf.to_float(tf.train.get_or_create_global_step())

    learning_rate *= (hidden_size ** -0.5)
    # Apply linear warmup
    learning_rate *= tf.minimum(1.0, step / warmup_steps)
    # Apply rsqrt decay
    learning_rate *= tf.rsqrt(tf.maximum(step, warmup_steps))

    tf.identity(learning_rate, "learning_rate")

    return learning_rate

def get_train_op_and_metrics(loss, params):
  """Generates training op and metrics to save in TensorBoard."""
  with tf.name_scope("get_train_op"):
    learning_rate = get_learning_rate(
            learning_rate=params["learning_rate"],
            hidden_size=params["hidden_size"],
            learning_rate_warmup_steps=params["learning_rate_warmup_steps"])

    optimizer = tf.contrib.opt.LazyAdamOptimizer(
            learning_rate,
            beta1=params["optimizer_adam_beta1"],
            beta2=params["optimizer_adam_beta2"],
            epsilon=params["optimizer_epsilon"])

    global_step = tf.train.get_global_step()
    tvars = tf.trainable_variables()
    gradients = optimizer.compute_gradients(
            loss, tvars, colocate_gradients_with_ops=True)
    minimize_op = optimizer.apply_gradients(
            gradients, global_step=global_step, name="train")
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    train_op = tf.group(minimize_op, update_op)

    train_metrics = {"learning_rate": learning_rate}

    gradient_norm = tf.global_norm(list(zip(*gradients))[0])
    train_metrics["global_norm/gradient_norm"] = gradient_norm

    return train_op, train_metrics

def translate_and_compute_bleu(estimator, subtokenizer, bleu_source, bleu_ref):
  """Translate file and report the cased and uncased bleu scores."""
  tmp = tmpfile.NamedTemporaryFile(delete=False)
  tmp_filename = tmp.name

  translate.translate_file(
          estimator, subtokenizer, bleu_source, output_file=tmp_filename,
          print_all_translations=False)

  # Compute uncased and cased belu scores.
  uncased_score = compute_bleu.bleu_wrapper(bleu_ref, tmp_filename, False)
  cased_score = compute_bleu.bleu_wrapper(bleu_ref, tmp_filename, True)
  os.remove(tmp_filename)
  return uncased_score, cased_score

def get_global_step(estimator):
  """Return estimator's last checkpoint."""
  return int(estimator.latest_checkpoint().split("-")[-1])

def evaluate_and_log_bleu(estimator, bleu_source, bleu_ref, vocab_file):
  """Calculates and records the BLEU score."""
  subtokenizer = tokenizer.Subtokenizer(vocab_file)

  uncased_score, cased_score = translate_and_compute_bleu(
          estimator, subtokenizer, bleu_source, bleu_ref)

  tf.logging.info("Bleu score (uncased): %d", uncased_score)
  tf.logging.info("Bleu score (cased): %d", cased_score)
  return uncased_score, cased_score

def _validate_file(filepath):
  """Make sure that file exists."""
  if not tf.gfile.Exists(filepath):
    raise tf.errors.NotFoundError(None, None, "File %s not found." % filepath)

def run_loop(
        estimator, schedule_manager, train_hooks=None, benchmark_logger=None,
        bleu_source=None, bleu_ref=None, bleu_threshold=None, vocab_file=None):
  """Train and evaluate model, and optionally compute model's BLEU score.

  **Setp vs. Epoch vs. Iteration**

  Steps and epochs are canonical terms used in TensorFlow and general machine
  learning. They are used to describe running a single process (train/eval):
    -Step refers to running the process through a single or batch of example.
    -Epoch refers to running the process through an entire dataset.
  
  E.g. training a dataset with 100 examples. The dataset is divided into 20
  batches with 5 examples per batch. A single training step trains the model
  on one batch. After 20 training steps, the model will have trained on every
  batch in the dataset, or, in other words, one epoch.

  Meanwhile, iteration is used in this implementation to describe running
  multiple processes (training and eval).
    - A single iteration:
      1. trains the model for a specific number of steps or epochs
      2. evaluates the model
      3. (if source and ref files are provided) compute BLEU score.
  
  This function runs through multiple train+eval+bleu iterations.

  Args:
    estimator: tf.Estimator containing model to train.
    schedule_manager: A schedule.Manager object to guide the run loop.
    train_hooks: List of hooks to pass to the estimator during training.
    benchmark_logger: a BenchmarkLogger object that logs evaluation data.
    bleu_source: File containing text to be translated for BLEU calculation.
    bleu_ref: File containing reference translation for BLEU calculation.
    bleu_threshold: minimum BLEU score before training is stopped.
    vocab_file: Path to vocab file that will be used to subtokenize bleu_score.

  Raises:
    ValueError: if both or none of single_iteration_train_steps and
      single_iteration_train_epochs were defined.
    NonFoundError: if the vocab file or bleu files don't exists.
  """
  if bleu_source:
    _validate_file(bleu_source)
  if bleu_ref:
    _validate_file(bleu_ref)
  if vocab_file:
    _validate_file(vocab_file)
  
  evaluate_bleu = bleu_source is not None and bleu_ref is not None
  
  # Print details of training schedule.
  tf.logging.info("Training schedule:")
  tf.logging.info(
    "\t1. Train for {}".format(schedule_manager.train_increment_str))
  tf.logging.info("\t2. Evaluate model.")
  if evaluate_bleu:
    tf.logging.info("\t3. Compute BLEU score.")
    if bleu_threshold is not None:
      tf.logging.info("Repeate above steps until the BLEU score reaches %f" %
              bleu_threshold)
  
  if not evaluate_bleu or bleu_threshold is None:
    tf.logging.info("Repeate above steps %d times." %
            schedule_manager.train_eval_iterations)

  if evaluate_bleu:
    # Create summary writer to log bleu score (values can be displayed in
    # Tensorboard).
    bleu_writer = tf.summary.FileWriter(
            os.path.join(estimator.model_dir, BLEU_DIR))
    if bleu_threshold is not None:
      # Change loop stopping condition if bleu_threshold is defined.
      schedule_manager.train_eval_iterations = INF

  # Loop training/evaluation/bleu cycles
  for i in xrange(schedule_manager.train_eval_iterations):
    tf.logging.info("Starting iteration %d" % (i + 1))

    # Train the model for single_iteration_train_steps or until the input fn
    # runs out of examples (if single_iteration_train_steps is None).
    estimator.train(
            dataset.train_input_fn,
            steps=schedule_manager.single_iteration_train_steps,
            hooks=train_hooks)

    eval_results = estimator.evaluate(
            input_fn=dataset.eval_input_fn,
            steps=schedule_manager.single_iteration_eval_steps)

    tf.logging.info("Evaluation results (iter %d/%d):" %
            (i + 1, schedule_manager.train_eval_iterations))
    tf.logging.info(eval_results)
    if benchmark_logger:
      benchmark_logger.log_evaluation_result(eval_results)

    # The results from estimator.evaluate() are measured on an approximate
    # translation, which utilize the target golden values provided. The actual
    # bleu score must be computed using the estimator.predict() path, which
    # outputs translations that are not based on golden values. The translations
    # are compared to reference file to get the actual bleu score.
    if evaluate_bleu:
      uncased_score, cased_score = evaluate_and_log_bleu(
              estimator, bleu_score, bleu_ref, vocab_file)

      # Write actual bleu scores using summary writer and benchmark logger
      global_step = get_global_step(estimator)
      summary = tf.Summary(value=[
          tf.Summary.Value(tag="bleu/uncased", simple_value=uncased_score),
          tf.Summary.Value(tag="bleu/cased", simple_value=cased_score),
        ])
      bleu_writer.add_summary(summary, global_step)
      bleu_writer.flush()
      if benchmark_logger:
        benchmark_logger.log_metric(
          "bleu_uncased", uncased_score, global_step=global_step)
        benchmark_logger.log_metric(
          "bleu_cased", cased_score, global_step=global_step)

      # Stop training if bleu stop threshold is defined.
      if model_helpers.past_stop_threshold(bleu_threshold, uncased_score):
        bleu_writer.close()
        break


def get_distribution_strategy(num_gpus, 
                              all_reduce_alg=None,
                              turn_off_distribution_strategy=False):
  """Return a DistributionStrategy for running the model.

  Args:
    num_gpus: Number of GPUs used.
    all_reduce_alg: Specify which algorithm to use when performing all-reduce.
      See tf.contrib.distribute.AllReduceCrossDeviceOps for available algorithms.
      If None, DistributionStrategy will choose based on device topology.
    turn_off_distribution_strategy: When set to True, do not use any 
      distribution strategy. Note that when it is True, and num_gpus is large 
      than 1, it will raise a ValueError.
  Returns:
    tf.contrib.distribute.DistributionStrategy object.
  Raises:
    ValueError: if turn_off_distribution_strategy is True and num_gpus is
    larger than 1.
  """
  if num_gpus == 0:
    if turn_off_distribution_strategy:
      return None
    else:
      return tf.contrib.distribute.OneDeviceStrategy("device:CPU:0")
  elif num_gpus == 1:
    if turn_off_distribution_strategy:
      return None
    else:
      return tf.contrib.distribute.OneDeviceStrategy("device:GPU:0")
  elif turn_off_distribution_strategy:
    raise ValueError("When {} GPUs are specified, "
            "turn_off_distribution_strategy flag cannot be set to "
            "'True'".format(num_gpus))
  else:
    if all_reduce_alg:
      return tf.contrib.distribute.AllReduceCrossDeviceOps(
              all_reduce_alg, num_packs=2)
    else:
      return tf.contrib.distribute.MirroredStrategy(num_gpus=num_gpus)
                              

def construct_estimator(params, config, schedule_manager):
  """Constructs an estimator from either Estimator or TPUEstimator.
  
  Args:
    params: The FLAGS parsed from command line.
    config: A dict of specific parameters.
    schedule_manager: A schedule.Manager object containing the run schedule.

  Returns:
    An estimator object to be used for training and eval.
  """
  distribution_strategy = get_distribution_strategy(
          params.num_gpus, params.all_reduce_spec)
  
  return tf.estimator.Estimator(
        model_fn=model_fn, model_dir=params.model_dir, params=config,
        config=tf.estimator.RunConfig(train_distribute=distribution_strategy))

class BenchmarkNLP(object):
  """Class for benchmarking a nlp network."""

  def __init__(self, params):
    """Initialize BenchmarkNLP.

    Args:
      params: Params tuple, created by make_params_from_flags.

    Raises:
      ValueError: Unsupported params settings.
    """
    self.params = params
    tf.logging.set_verbosity(tf.logging.INFO)
  
  def run(self): 
    """Run the particular NLP benchmark depend on given model."""
    if self.params.model == 'bert':
      self.run_bert_model()
    elif self.params.model == 'transformer':
      self.run_transformer_model()
    else:
      raise ValueError("Unsupported model: %s", self.params.model)

  def run_transformer_model(self):
    """Run the transformer benchmarks."""
    num_gpus = self.params.num_gpus
    config = PARAMS_MAP[self.params.param_set]
    if num_gpus > 1:
      if self.params.param_set == "big":
        config = model_params.BIG_MULTI_GPU_PARAMS
      else:
        config = model_params.BASE_MULTI_GPU_PARAMS

    config['data_dir'] = self.params.data_dir
    config['model_dir'] = self.params.model_dir
    config['num_parallel_calls'] = self.params.num_parallel_calls
    config['static_batch'] = self.params.static_batch
    config['allow_ffn_pad'] = True
    config['use_synthetic_data'] = self.params.use_synthetic_data
    config['batch_size'] = self.params.batch_size
    tf.logging.info("batch size per device: {}, total batch size: {}".format(
                    self.params.batch_size, self.params.batch_size * num_gpus))

    schedule_manager = schedule.Manager(
            train_steps=self.params.train_steps,
            steps_between_evals=self.params.steps_between_evals,
            train_epochs=self.params.train_epochs,
            epochs_between_evals=self.params.epochs_between_evals,
            default_train_epochs=DEFAULT_TRAIN_EPOCHS,
            batch_size=config['batch_size'],
            max_length=config['max_length'])

    config['repeat_dataset'] = schedule_manager.repeat_dataset

    # Train and evaluate transformer model
    estimator = construct_estimator(self.params, config, schedule_manager)
    run_loop(
            estimator=estimator,
            # Training arguments
            schedule_manager=schedule_manager,
            train_hooks=None,
            benchmark_logger=None,
            # BLEU calculation arguments
            bleu_source=self.params.bleu_source,
            bleu_ref=self.params.bleu_ref,
            bleu_threshold=None,
            vocab_file=self.params.vocab_file)
    tf.logging.info("Completed transformer benchmark.")

  def run_bert_model(self):
    """Run the BERT benchmark task assigned to this process."""
    processors = {
        "cola": ColaProcessor,
        "mnli": MnliProcessor,
        "mrpc": MrpcProcessor,
        "xnli": XnliProcessor,
    }

    tokenization.validate_case_matches_checkpoint(self.params.do_lower_case,
                                                  self.params.init_checkpoint)

    if not self.params.do_train and not self.params.do_eval and (
                  not self.params.do_predict):
      raise ValueError("At least one of 'do_train', 'do_eval' or 'do_predict' "
                       "must be true.")

    bert_config = bert_model.BertConfig.from_json_file(
            self.params.bert_config_file)

    if self.params.max_seq_length > bert_config.max_position_embeddings:
      raise ValueError("Cannot use sequence length %d because the BERT model "
                       "was only trained up to sequence length %d" %
                       (self.params.max_seq_length, 
                           bert_config.max_position_embeddings))

    tf.gfile.MakeDirs(self.params.output_dir)

    task_name= self.params.task_name
    if task_name not in processors:
      raise ValueError("Task '%s' not found." % (task_name))

    processor = processors[task_name]()
    label_list = processor.get_labels()

    tokenizer = tokenization.FullTokenizer(
            vocab_file=self.params.vocab_file,
            do_lower_case=self.params.do_lower_case)

    tpu_cluster_resolver = None
    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
            cluster=tpu_cluster_resolver,
            master=self.params.master,
            model_dir=self.params.output_dir,
            save_checkpoints_steps=self.params.save_checkpoints_steps,
            tpu_config=tf.contrib.tpu.TPUConfig(
                iterations_per_loop=self.params.iterations_per_loop,
                num_shards=None,
                per_host_input_for_training=is_per_host))
    
    train_examples = None
    num_train_steps = None
    num_warmup_steps = None

    if self.params.do_train:
      train_examples = processor.get_train_examples(self.params.data_dir)
      num_train_steps = int(
            len(train_examples) / self.params.batch_size * (
            self.params.num_train_epochs))
      num_warmup_steps = int(num_train_steps * self.params.warmup_proportion)

    model_fn = model_fn_builder(
            bert_config=bert_config,
            num_labels=len(label_list),
            init_checkpoint=self.params.init_checkpoint,
            learning_rate=self.params.learning_rate,
            num_train_steps=num_train_steps,
            num_warmup_steps=num_warmup_steps)

    # If TPU is not available, this will fall back to noraml Estimator on CPU
    # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
            use_tpu=False,
            model_fn=model_fn,
            config=run_config,
            train_batch_size=self.params.batch_size,
            eval_batch_size=self.params.eval_batch_size,
            predict_batch_size=self.params.predict_batch_size)

    if self.params.do_train:
      train_file = os.path.join(self.params.output_dir, "train.tf_record")
      file_based_convert_examples_to_features(
            train_examples, label_list, self.params.max_seq_length, tokenizer,
            train_file)
      tf.logging.info("***** Running training *****")
      tf.logging.info("  Number examples = %d", len(train_examples))
      tf.logging.info("  Batch size = %d", self.params.batch_size)
      tf.logging.info("  Num steps = %d", num_train_steps)
      train_input_fn = file_based_input_fn_builder(
              input_file=train_file,
              seq_length=self.params.max_seq_length,
              is_training=True,
              drop_remainder=True)
      estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    if self.params.do_eval:
      eval_examples = processor.get_dev_examples(self.params.data_dir)
      num_actual_eval_examples = len(eval_examples)

      eval_file = os.path.join(self.params.output_dir, "eval.tf_record")
      file_based_convert_examples_to_features(
              eval_examples, label_list, self.params.max_seq_length, tokenizer,
              eval_file)
      tf.logging.info("***** Running evaluation *****")
      tf.logging.info("  Number examples = %d (%d actual, %d padding)", 
              len(eval_examples), num_actual_eval_examples,
              len(eval_examples) - num_actual_eval_examples)
      tf.logging.info("  Batch size = %d", self.params.eval_batch_size)

      eval_input_fn = file_based_input_fn_builder(
              input_file=eval_file,
              seq_length=self.params.max_seq_length,
              is_training=False,
              drop_remainder=False)

      result = estimator.evaluate(input_fn=eval_input_fn, steps=None)

      output_eval_file = os.path.join(self.params.output_dir, "eval_results.txt")
      with tf.gfile.GFile(output_eval_file, "w") as writer:
        tf.logging.info("***** Eval results *****")
        for key in sorted(result.keys()):
          tf.logging.info("  %s = %s", key, str(result[key]))
          writer.write("%s = %s\n" % (key, str(result[key])))

    if self.params.do_predict:
      predict_examples = processor.get_test_examples(self.params.data_dir)
      num_actual_predict_examples = len(predict_examples)

      predict_file = os.path.join(self.params.out_dir, "predict.tf_record")
      file_based_convert_examples_to_features(
              predict_examples, label_list, self.params.max_seq_length,
              tokenizer, predict_file)

      tf.logging.info("***** Running predication *****")
      tf.logging.info("  Number examples = %d (%d actual, %d padding)",
              len(predict_examples), num_actual_predict_examples,
              len(predict_examples) - num_actual_predict_examples)
      tf.logging.info("  Batch size = %d", self.params.batch_size)

      predict_input_fn = file_based_input_fn_builder(
              input_file=predict_file,
              seq_length=self.params.max_seq_length,
              is_training=False,
              drop_remainder=False)

      result = estimator.predict(input_fn=predict_input_fn)

      output_predict_file = os.path.join(self.params.output_dir, 
              "test_results.txt")
      with tf.gfile.GFile(output_predict_file, "w") as writer:
        num_written_lines = 0
        tf.logging.info("***** Predict results *****")
        for (i, prediction) in enumerate(result):
          probabilities = prediction["probabilities"]
          if i >= num_actual_predict_examples:
            break
          output_line = "\t".join(
                  str(class_probability)
                  for class_probability in probabilities) + "\n"
          writer.write(output_line)
          num_written_line += 1
      assert num_written_lines == num_actual_predict_examples
