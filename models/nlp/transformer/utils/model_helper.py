"""
Functions and classes for transformer model.
"""
from models.nlp.transformer.model import model_params
from models.nlp.transformer.model import schedule
from models.nlp.transformer.utils import metrics
from models.nlp.transformer.utils import optimization
from models.nlp.transformer.utils import dataset
from models.nlp.transformer.model import transformer

import tensorflow as tf

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
              mode=mode,
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
              mode=mode, loss=loss, predictions={'predictions': logits})
    else:
      train_op = optimization.create_optimizer(loss, 
              params["learning_rate"],
              params["hidden_size"],
              params["num_train_steps"],
              params["warmup_step"])

      return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

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
        bleu_source=None, bleu_ref=None, vocab_file=None):
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
            steps=10,
            #steps=schedule_manager.single_iteration_train_steps,
            hooks=None)
            #hooks=train_hooks)

    tf.logging.info("Evaluation.")

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
  """Constructs an estimator from Estimator.
  
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
        model_fn=model_fn, model_dir=params.output_dir, params=config,
        config=tf.estimator.RunConfig(train_distribute=distribution_strategy))


