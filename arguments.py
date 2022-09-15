import argparse
from pydoc import description


def add_model_args(parser: argparse.ArgumentParser):
    """Add arguments for model configs.
    """
    group = parser.add_argument_group(
        title="model", description="model configs")
    group.add_argument('--model-name',  default=None,   type=str,
                       help='Path to pretrained teacher model or model identifier from huggingface.co/models')
    group.add_argument('--config-name', default=None,   type=str,
                       help='Pretrained config name or path if not the same as model_name_or_path')
    group.add_argument('--tokenizer-name',  default=None,   type=str,
                       help='Pretrained tokenizer name or path if not the same as model_name_or_path')
    group.add_argument('--cache-dir',   default=None,   type=str,
                       help='Where do you want to store the pretrained models downloaded from huggingface.co')
    group.add_argument('--model-revision', default='main', type=str,
                       help='The specific model version to use (can be a branch name, tag name or commit id).')
    return parser


def add_data_args(parser: argparse.ArgumentParser):
    """Add arguments for data configs.
    """
    group = parser.add_argument_group(
        title='data', description="data configs")
    group.add_argument('--dataset', default=None, type=str,
                       help='The name of the dataset to use (via the datasets library).')
    group.add_argument('--max-seq-length', default=128, type=int,
                       help='The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded if `--pad_to_max_length` is passed.')
    group.add_argument('--pad-to-max-length', action='store_true',
                       help='Whether to pad all samples to `max_seq_length`. If False, will pad the samples dynamically when batching to the maximum length in the batch.')
    group.add_argument('--overwrite-cache', action='store_true',
                       help='Overwrite the cached preprocessed datasets or not.')
    group.add_argument('--max-train-samples', default=None, type=int,
                       help="For debugging purposes or quicker training, truncate the number of training examples to this value if set.")
    group.add_argument('--max-eval-samples', default=None, type=int,
                       help="For debugging purposes or quicker training, truncate the number of evaluation examples to this value if set.")
    group.add_argument('--max-prefict-samples', default=None, type=int,
                       help="For debugging purposes or quicker training, truncate the number of prediction examples to this value if set.")
    group.add_argument('--train-file', default=None, type=str,
                       help='A csv or a json file containing the training data.')
    group.add_argument('--valid-file', default=None, type=str,
                       help="A csv or a json file containing the validation data.")
    group.add_argument('--test-file', default=None, type=str,
                       help="A csv or a json file containing the test data.")
    return parser


def add_training_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(
        title='training', description='training configs')
    group.add_argument('--model-type', default=None, type=str,
                       help="Type of model head: Sequence Classification (cls), Question Answering (qa) or Token Classification (tcls).")
    group.add_argument('--load', type=str, default=None,
                       help='Path to a directory containing a model checkpoint.')
    group.add_argument('--save', type=str, default=None,
                       help='Output directory to save checkpoints to.')
    group.add_argument('--save-name', type=str, default=None,
                       help='Output filename to save checkpoints to.')
    group.add_argument('--do-train', action='store_true',
                       help='Whether to run training.')
    group.add_argument('--do-eval', action='store_true',
                       help='Whether to run eval on the dev set.')
    group.add_argument('--do-predict', action='store_true',
                       help='Whether to run predictions on the test set.')
    group.add_argument('--train-batch-size', type=int, default=32,
                       help='Data Loader batch size while training.')
    group.add_argument('--epoch-num', type=int, default=20,
                       help='Total number of epochs.')
    group.add_argument('--seed', type=int, default=1234,
                       help='Random seed for reproducibility.')
    group.add_argument('--lr', type=float, default=1e-4,
                       help='Initial learning rate')
    group.add_argument('--eval-batch-size', type=int, default=16,
                       help='batch size in evaluation')
    group.add_argument('--log-dir', type=str, default=None,
                       help='tensorboard log directory')
    group.add_argument('--output-dir', type=str, default=None,
                       help='output file')
    group.add_argument('--log-level', type=str, default='passive',
                       help="Logger log level to use on the main process. Possible choices are the log levels as strings: 'debug', 'info', 'warning', 'error' and 'critical', plus a 'passive' level which doesn't set anything and lets the application set the level.")
    return parser


def add_special_args(parser: argparse.ArgumentParser):
    """Special arguments"""

    group = parser.add_argument_group(
        title='special', description='special configurations')

    # Inference configurations
    group.add_argument_group(
        title='inference', description='inference configurations')
    group.add_argument('--beam-size', type=int, default=1,
                       help='beam size')
    group.add_argument('--top-k', type=int, default=0,
                       help='top k')
    group.add_argument('--top-p', type=float, default=0.0,
                       help='top p')
    group.add_argument('--temperature', type=float, default=0.9,
                       help='temperature')
    group.add_argument('--no-repeat-ngram-size', type=int, default=0,
                       help='no repeat ngram size')
    group.add_argument('--repetition-penalty', type=float, default=1.0,
                       help='repetition penalty')
    group.add_argument('--random-sample', default=False, action='store_true',
                       help='use random sample strategy')
    group.add_argument('--use-contrastive-search', default=False, action='store_true',
                       help='whether to use contrastive search')
    return parser


def get_args():
    """Prepare arguments for training and evaluation.
    """
    parser = argparse.ArgumentParser()
    parser = add_data_args(parser)
    parser = add_model_args(parser)
    parser = add_training_args(parser)
    # parser = add_special_args(parser)
    args = parser.parse_args()

    return args
