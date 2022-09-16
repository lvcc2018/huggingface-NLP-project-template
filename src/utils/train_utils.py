import numpy as np
import sys
import logging
import random
import torch

from transformers import utils
from transformers import get_linear_schedule_with_warmup

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForConditionalGeneration,
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    AutoModelForTokenClassification,
)


from src.consts import *


def get_model_obj(args):
    if args.model_type == CG:
        return AutoModelForConditionalGeneration
    elif args.model_type == CLS:
        return AutoModelForSequenceClassification
    elif args.model_type == QA:
        return AutoModelForTokenClassification
    elif args.model_type == TCLS:
        return AutoModelForQuestionAnswering
    else:
        raise ValueError(
            f"Model type {args.model_type} is not supported. Available types are {ALL_MODEL_TYPES}")


def set_logging(args, logger):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = args.log_level
    logger.setLevel(log_level)
    utils.logging.set_verbosity(log_level)
    utils.logging.enable_default_handler()
    utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.info(f"Training configs: {args}")


def set_seed(args):
    """Set the seed for reproducibility."""

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        cache_dir=args.cache_dir
    )
    return tokenizer


def get_model(args):
    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir)
    model = get_model_obj(args).from_pretrained(
        args.model_name_or_path,
        config=config,
        cache_dir=args.cache_dir)
    return model


def get_optimizer(args, model):
    return torch.optim.AdamW(model.parameters(), lr=args.learning_rate)


def get_learning_rate_scheduler(args, optimizer):
    return get_linear_schedule_with_warmup(optimizer)


def set_model_and_optimizer(args):
    tokenizer = get_tokenizer(args)
    model = get_model(args)
    optimizer = get_optimizer(args, model)
    scheduler = get_learning_rate_scheduler(args, optimizer)
    return tokenizer, model, optimizer, scheduler
