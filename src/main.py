from lib2to3.pgen2 import token
import logging
import os
from pickletools import optimize
import random
import sys
from tqdm import tqdm

from consts import *
from arguments import get_args

from utils.data_utils import *
from utils.train_utils import *


logger = logging.getLogger(__name__)


def train_model(args, raw_datasets):

    # Load pretrained model and tokenizer, prepare optimizer and schedule
    tokenizer, model, optimizer, scheduler = set_model_and_optimizer(args)

    # Prepare datasets and dataloaders
    train_dataset, valid_dataset, test_dataset = preprocess_datasets(
        args, raw_datasets, tokenizer, logger)
    train_dataloader, valid_dataloader, test_dataloader = get_dataloaders(
        args, train_dataset, valid_dataset, test_dataset, tokenizer)

    # Log a few random samples from the training set:
    if args.do_sample:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(
                f"Sample {index} of the training set: {train_dataset[index]}.")

    # TODO:Prepare metrics

    # TODO:Training
    if args.do_train:
        pass

    # TODO:Evaluation
    if args.do_eval:
        pass

    # TODO:Prediction
    if args.do_predict:
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


if __name__ == "__main__":
    main()
