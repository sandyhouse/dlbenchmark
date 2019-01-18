"""The main program for Deep Learning Benchmarks"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app as absl_app
from absl import flags

def run_benchmark(_):
    """Run the benchmark."""
    if FLAGS.model == "alexnet_tf_keras":
        from models.tensorflow.keras import alexnet
        alexnet_model = alexnet.AlexnetModel(FLAGS)
        alexnet_model.build_network(FLAGS)
        alexnet_model.train(FLAGS)


MODELS = ["alexnet_tf_keras"]
DISTRIBUTE_STRATEGIES = ['MirroredStrategy', 'CollectiveAllReducestrategy',
            'ParameterServerStrategy']

# Define benchmark flags.
def define_benchmark_flags():
    flags.DEFINE_enum(
        name="model", default=None,
        enum_values=MODELS,
        help="Model to be benchmarked.\n"
             "Use --list_models to show all available models.")
    flags.DEFINE_bool(
        name="list_models", default=False,
        help="Show all available models.")
    flags.DEFINE_integer(
        name="num_gpus", default=1,
        help="Number of GPUs to use, set 0 to use CPUs only.")
    flags.DEFINE_string(
        name="data_dir", default='/tmp/data_dir',
        help="Directory for data used for train and evaluation.")
    flags.DEFINE_integer(
        name="batch_size", default=128,
        help="Batch size for train.")
    flags.DEFINE_integer(
        name="epoch_size", default=10,
        help="The number of epoches to train.")
    flags.DEFINE_enum(
        name="distribute_strategy", default=None,
        enum_values=DISTRIBUTE_STRATEGIES,
        help="Distribute strategy to use.\n"
             "Use --list-distribute-strategies to show "
             "all available distribute strategies.")
    flags.DEFINE_bool(
        name="list_distribute_strategies", default=False,
        help="Show all available distribute strategies.")
    flags.DEFINE_float(
        name="learning_rate", default=0.01,
        help="The initial value for learning rate.")
    flags.DEFINE_enum(
        name="data_type", default='float32',
        enum_values=['float32', 'float16'],
        help="The data type used.")
    #flags.mark_flag_as_required('model')
    #flags.mark_flag_as_required('distribution_strategy')

def main(_):
    if FLAGS.list_models:
        print("All models available for the benchmark:\n")
        for model in MODELS:
            print('\t', model)
        return
    if FLAGS.list_distribute_strategies:
        print("All distribute strategies available for the benchmark:\n")
        for value in DISTRIBUTE_STRATEGIES:
            print('\t', value)
        return
    if FLAGS.model == None:
        print("The model should be specified by --model.\n"
              "Use --list_models to show all available models.")
        return
    if FLAGS.distribute_strategy == None:
        print("The distribute strategy should be specified by "
              "--distribute_strategy.\n"
              "Use --list_distribute_strategies to show all available models.")
        return
                                          
    run_benchmark(FLAGS)


if __name__ == '__main__':
    define_benchmark_flags()
    FLAGS = flags.FLAGS
    absl_app.run(main)
