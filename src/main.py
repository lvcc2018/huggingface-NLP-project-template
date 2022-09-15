import logging
import os
from pickletools import optimize
import random
import sys

from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorWithPadding,
    HfArgumentParser,
    default_data_collator,
    set_seed,
)
from transformers.utils.versions import require_version

from consts import *
from arguments import get_args

from utils.data_utils import *
from utils.train_utils import *



logger = logging.getLogger(__name__)


def train_model(args, raw_datasets, iteration=0):

    # Load pretrained model and tokenizer
    tokenizer = get_tokenizer(args)
    model = get_model(args)

    # Tokenize datasets
    train_dataset, valid_dataset, test_dataset = preprocess_datasets(args, raw_datasets)

    # Log a few random samples from the training set:
    if args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Prepare optimizer and schedule
    optimizer = get_optimizer(args, model)
    scheduler = get_learning_rate_scheduler(args, optimizer)

    # TODO:Prepare metrics

    # TODO:Training
    if args.do_train:
        pass

    # TODO:Evaluation
    if args.do_eval:
        pass



def main():
    # Get the arguments from the command line
    args = get_args()

    # Setup logging
    set_logging(args, logger)
    

    # Set seed before initializing model.
    set_seed(args)

    # Load datasets
    raw_datasets = load_dataset_from_files(args.dataset)
    

    # run training
    trainer = train_model(args, args, args, raw_datasets)


if __name__ == "__main__":
    main()
