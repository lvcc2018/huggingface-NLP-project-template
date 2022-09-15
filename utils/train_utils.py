import numpy as np
import sys
import logging
import random
import torch

from transformers import utils
from datasets import load_metric

from transformers import (
    AutoModelForConditionalGeneration,
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    AutoModelForTokenClassification,
    EvalPrediction
)

from datasets import load_metric

from consts import *


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


def get_compute_metrics(metrics):
    # Get the metric functions
    metrics = {metric: load_metric(metric) for metric in metrics}

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(
            p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        result = {metric: metric_fn.compute(predictions=preds, references=p.label_ids)[
            metric] for metric, metric_fn in metrics.items()}
        return result

    return compute_metrics


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
