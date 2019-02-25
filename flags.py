# -*- coding:utf-8 -*-
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import multiprocessing

parser = argparse.ArgumentParser(
        usage='%(prog)s [options]',
        description="Benchmarks for Deep Learning Platform - TensorFlow.")

# Define commond line arguments for benchmarks
def define_flags():
  parser.add_argument("--backend",
                      choices=['TensorFlow', 'PyTorch', 
                               'tensorflow', 'pytorch'],
                      default="TensorFlow",
                      help="The backend framework used for benchmarks.")
  parser.add_argument("--model",
                      type=str,
                      default=None,
                      help="Name of the model to run.")
  parser.add_argument("--batch_size",
                      type=int,
                      default=32,
                      help="Batch size per computing device.")
  parser.add_argument("--num_epochs",
                      type=float,
                      default=1,
                      help="Number of epochs to run.")
  parser.add_argument("--num_gpus",
                      type=int,
                      default=1,
                      help="Number of GPUs to use. If the value is set to 0, "
                           "CPUs will be used instead.")
  parser.add_argument("--data_dir",
                      type=str,
                      default=None,
                      help="Directory for input dataset. If it is not set, "
                           "synthetic data will be used.")
  parser.add_argument("--do_train",
                      default=False,
                      action='store_true',
                      help="Whether to run training.")
  parser.add_argument("--do_eval",
                      default=False,
                      action='store_true',
                      help="Whether to run evaluation.")
  parser.add_argument("--do_predict",
                      default=False,
                      action='store_true',
                      help="Whether to run prediction.")
  parser.add_argument("--data_format",
                      choices=['NHWC', 'NCHW'],
                      default="NCHW",
                      help="Data format to use. Recommendation: NHWC for CPUs, "
                           "and NCHW for GPUs.")
  parser.add_argument("--optimizer",
                      choices=['momentum', 'sgd', 'adam'],
                      default="adam",
                      help="Optimizer to use.")
  parser.add_argument("--init_learning_rate",
                      type=float,
                      default=0.1,
                      help="Initial learning rate to use.")
  parser.add_argument("--momentum",
                      type=float,
                      default=0.9,
                      help="Momentum for training.")
  parser.add_argument("--adam_beta1",
                      type=float,
                      default=0.9,
                      help="Beta1 term for Adam optimizer.")
  parser.add_argument("--adam_beta2",
                      type=float,
                      default=0.999,
                      help="Beta2 term for Adam optimizer.")
  parser.add_argument("--adam_epsilon",
                      type=float,
                      default=1e-8,
                      help="Epsilon term for Adam optimizer.")
  parser.add_argument("--use_fp16",
                      default=False,
                      action='store_true',
                      help="Whether to use float16 for tensors.")
  parser.add_argument("--all_reduce_spec",
                      type=str,
                      default=None,
                      help="A specification of the all_reduce algorithm to use "
                           "for reducing gradients.")
  parser.add_argument("--save_checkpoints_steps",
                      type=int,
                      default=None,
                      help="How often to save trained models.")
  parser.add_argument("--ip_list",
                      type=str,
                      default=None,
                      help="iplist imported by mpirun (for slurm).")
  parser.add_argument("--job_name",
                      type=str,
                      default=None,
                      help="Name of the job, e.g., 'worker', 'ps', etc.")
  parser.add_argument("--job_index",
                      type=int,
                      default=None,
                      help="Index of the job, e.g., 0, 1, etc.")
  parser.add_argument("--max_chkpts_to_keep",
                      type=int,
                      default=5,
                      help="Maximum number of checkpoints to keep.")
  parser.add_argument("--model_dir",
                      type=str,
                      default='/tmp',
                      help="Directory where model checkpoints will be stored.")
  parser.add_argument("--init_checkpoint",
                      type=str,
                      default=None,
                      help="Path for the initial model checkpoint.")

# Flags for NLP models
  parser.add_argument("--vocab_file",
                     type=str,
                     default=None,
                     help="The vocabulary file to train on.")
  parser.add_argument("--max_seq_length",
                     type=int,
                     default=128,
                     help="Maximum length of input sequence. Sequences longer "
                          "than that will be truncated, and sequences shorter "
                          "than that will be padded.")
  parser.add_argument("--param_set",
                      type=str,
                      default="big",
                      help="Parameter set to use when creating and training "
                           "the model. The parameters define the input shape "
                           "(batch batch_size and max length), model "
                           "configuration (size of embedding, # of hidden "
                           "layers, etc.), and various other settings. "
                           "The big parameter set increases the default batch "
                           "size, embedding/hidden size, and filter size.")
  parser.add_argument("--bleu_source",
                      type=str,
                      default=None,
                      help="Path to source file containing text translate when "
                           "calculating the official BLEU score.")
  parser.add_argument("--bleu_ref",
                      type=str,
                      default=None,
                      help="Path to source file containing text translate when "
                           "calculating the official BLEU score.")

  parser.add_argument("--bert_config_file",
                     type=str,
                     default=None,
                     help="The config file (json) for the BERT model, which "
                          "specifies the model architecture.")
  parser.add_argument("--task_name",
                     type=str,
                     default=None,
                     help="The name of the task to train.")
  parser.add_argument("--do_lower_case",
                     default=True,
                     action='store_true',
                     help="Whether to use low case for input text.")
  parser.add_argument("--train_file",
                     type=str,
                     default=None,
                     help="SQuAD json for training. E.g., train-v1.1.json.")
  parser.add_argument("--predict_file",
                     type=str,
                     default=None,
                     help="SQuAD json for predictions. E.g., dev-v1.1.json.")
  parser.add_argument("--doc_stride",
                     type=int,
                     default=128,
                     help="When splitting up a long document into chunks, how "
                          "much stride to taken between chunks.")
  parser.add_argument("--max_query_length",
                     type=int,
                     default=64,
                     help="The maximum number of tokens for the question. "
                          "Questions longer than this will be truncated to "
                          "this length.")
  parser.add_argument("--n_best_size",
                     type=int,
                     default=20,
                     help="The total number of n-best predictions to generate "
                          "in the nbest_predictions.json output file.")
  parser.add_argument("--max_answer_length",
                     type=int,
                     default=30,
                     help="The maximum length of an answer that can be "
                          "generated. This is needed because the start and end "
                          "predictions are not conditioned on one another.")
  parser.add_argument("--version_2_with_negative",
                     default=False,
                     action='store_true',
                     help="If true, the SQuAD examples contain some that do "
                          "not have an answer.")
  parser.add_argument("--run_squad",
                     default=False,
                     action='store_true',
                     help="If true, run SQuAD tasks, otherwise run sequence "
                          "(sequence-pair) classification task.")
