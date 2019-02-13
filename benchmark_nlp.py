"""TensorFlow benchmark library for NLP"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os
from models.nlp.bert.model import bert_model
from models.nlp.bert.utils import optimization
from models.nlp.bert.utils import tokenization
from models.nlp.bert.utils import model_helper as bert_helper
from models.nlp.transformer.model import model_params
from models.nlp.transformer.model import schedule
from models.nlp.transformer.utils import metrics
from models.nlp.transformer.utils import dataset
from models.nlp.transformer.model import transformer
from models.nlp.transformer.utils import model_helper as transformer_helper
import tensorflow as tf

PARAMS_MAP = {
        'tiny': model_params.TINY_PARAMS,
        'base': model_params.BASE_PARAMS,
        'big': model_params.BIG_PARAMS,
}

DEFAULT_TRAIN_EPOCHS = 10

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
      raise ValueError("model: %s is not implemented.", self.params.model)

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
    config['model_dir'] = self.params.output_dir
    config['num_parallel_calls'] = self.params.num_parallel_calls
    config['static_batch'] = self.params.static_batch
    config['allow_ffn_pad'] = True
    config['use_synthetic_data'] = self.params.use_synthetic_data
    config['batch_size'] = self.params.batch_size
    total_batch_size = 0
    if self.params.num_gpus == 0:
      total_batch_size = self.params.batch_size
    else:
      total_batch_size = self.params.batch_size * num_gpus
    tf.logging.info("batch size per device: {}, total batch size: {}".format(
                    self.params.batch_size, total_batch_size))

    schedule_manager = schedule.Manager(
            train_steps=self.params.num_batches,
            steps_between_evals=10,
            train_epochs=self.params.num_epochs,
            epochs_between_evals=1,
            default_train_epochs=DEFAULT_TRAIN_EPOCHS,
            batch_size=config['batch_size'],
            max_length=config['max_length'])

    config['repeat_dataset'] = schedule_manager.repeat_dataset

    # Train and evaluate transformer model
    estimator = transformer_helper.construct_estimator(self.params, config, 
            schedule_manager)
    transformer_helper.run_loop(
            estimator=estimator,
            # Training arguments
            schedule_manager=schedule_manager,
            train_hooks=None,
            benchmark_logger=None,
            # BLEU calculation arguments
            bleu_source=self.params.bleu_source,
            bleu_ref=self.params.bleu_ref,
            vocab_file=self.params.vocab_file)
    tf.logging.info("Completed transformer benchmark.")

  def run_bert_model(self):
    """Run the BERT benchmark task assigned to this process."""
    processors = {
        "cola": bert_helper.ColaProcessor,
        "mnli": bert_helper.MnliProcessor,
        "mrpc": bert_helper.MrpcProcessor,
        "xnli": bert_helper.XnliProcessor,
    }

    tokenization.validate_case_matches_checkpoint(self.params.do_lower_case,
                                                  self.params.init_checkpoint)

    if not self.params.do_train and not self.params.do_eval and (
                  not self.params.do_predict):
      raise ValueError("At least one of 'do_train', 'do_eval' or 'do_predict' "
                       "must be set.")

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

    run_config = tf.estimator.RunConfig(
            model_dir=self.params.output_dir,
            save_checkpoints_steps=self.params.save_checkpoints_steps)
    
    train_examples = None
    num_train_steps = None
    num_warmup_steps = None

    if self.params.do_train:
      train_examples = processor.get_train_examples(self.params.data_dir)
      if self.params.num_epochs:
        num_train_steps = int(
            len(train_examples) / self.params.batch_size * (
                self.params.num_epochs))
      elif self.params.num_batches:
        num_train_steps = self.params.num_batches
      else:
        num_train_steps = int(len(train_examples) / self.params.batch_size)
      num_warmup_steps = (int(self.params.num_warmup_batches) 
            if self.params.num_warmup_batches else 0)

    model_fn = bert_helper.model_fn_builder(
            bert_config=bert_config,
            num_labels=len(label_list),
            init_checkpoint=self.params.init_checkpoint,
            learning_rate=self.params.init_learning_rate,
            num_train_steps=num_train_steps,
            num_warmup_steps=num_warmup_steps)

    estimator = tf.estimator.Estimator(
            model_fn=model_fn,
            config=run_config)

    if self.params.do_train:
      train_file = os.path.join(self.params.output_dir, "train.tf_record")
      bert_helper.file_based_convert_examples_to_features(
            train_examples, label_list, self.params.max_seq_length, tokenizer,
            train_file)
      tf.logging.info("***** Running training *****")
      tf.logging.info("  Number examples = %d", len(train_examples))
      tf.logging.info("  Batch size = %d", self.params.batch_size)
      tf.logging.info("  Num steps = %d", num_train_steps)
      train_input_fn = bert_helper.file_based_input_fn_builder(
              input_file=train_file,
              seq_length=self.params.max_seq_length,
              is_training=True,
              drop_remainder=True,
              batch_size=self.params.batch_size)
      estimator.train(input_fn=train_input_fn, steps=num_train_steps)

    if self.params.do_eval:
      eval_examples = processor.get_dev_examples(self.params.data_dir)
      num_actual_eval_examples = len(eval_examples)

      num_eval_steps = int(num_actual_eval_examples / self.params.batch_size)

      eval_file = os.path.join(self.params.output_dir, "eval.tf_record")
      bert_helper.file_based_convert_examples_to_features(
              eval_examples, label_list, self.params.max_seq_length, tokenizer,
              eval_file)
      tf.logging.info("***** Running evaluation *****")
      tf.logging.info("  Number examples = %d (%d actual, %d padding)", 
              len(eval_examples), num_actual_eval_examples,
              len(eval_examples) - num_actual_eval_examples)
      tf.logging.info("  Batch size = %d", self.params.batch_size)

      eval_input_fn = bert_helper.file_based_input_fn_builder(
              input_file=eval_file,
              seq_length=self.params.max_seq_length,
              is_training=False,
              drop_remainder=False,
              batch_size=self.params.batch_size)

      result = estimator.evaluate(input_fn=eval_input_fn,
                                  steps=num_eval_steps)

      output_eval_file = os.path.join(self.params.output_dir, 
                                      "eval_results.txt")
      with tf.gfile.GFile(output_eval_file, "w") as writer:
        tf.logging.info("***** Eval results *****")
        for key in sorted(result.keys()):
          tf.logging.info("  %s = %s", key, str(result[key]))
          writer.write("%s = %s\n" % (key, str(result[key])))

    if self.params.do_predict:
      predict_examples = processor.get_test_examples(self.params.data_dir)
      num_actual_predict_examples = len(predict_examples)

      predict_file = os.path.join(self.params.output_dir, "predict.tf_record")
      bert_helper.file_based_convert_examples_to_features(
              predict_examples, label_list, self.params.max_seq_length,
              tokenizer, predict_file)

      tf.logging.info("***** Running predication *****")
      tf.logging.info("  Number examples = %d (%d actual, %d padding)",
              len(predict_examples), num_actual_predict_examples,
              len(predict_examples) - num_actual_predict_examples)
      tf.logging.info("  Batch size = %d", self.params.batch_size)

      predict_input_fn = bert_helper.file_based_input_fn_builder(
              input_file=predict_file,
              seq_length=self.params.max_seq_length,
              is_training=False,
              drop_remainder=False,
              batch_size=self.params.batch_size)

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
          num_written_lines += 1
      assert num_written_lines == num_actual_predict_examples
