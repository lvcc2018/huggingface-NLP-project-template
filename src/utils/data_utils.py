from cgi import test
from datasets import load_dataset
from src.consts import *
from transformers import AutoTokenizer


def load_dataset_from_files(args, suffix=JSON):
    data_dir = DATA_DIR / args.dataset
    data_files = {
        TRAIN: data_dir / args.train_file,
        VALID: data_dir / args.valid_file,
        TEST: data_dir / args.test_file
    }
    datasets = load_dataset(suffix, data_files=data_files)
    return datasets


def preprocess_datasets(args, raw_datasets, tokenizer, logger):

    # Padding strategy
    if args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    if args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(instance):
        text = instance['text']
        result = tokenizer(text, padding=padding,
                           max_length=max_seq_length, truncation=True)

        if "label" in instance:
            result["label"] = instance["label"]
        return result

    tokenized_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        load_from_cache_file=not args.overwrite_cache,
        desc="Running tokenizer on dataset",
    )

        # Make sure datasets are here and select a subset if specified
    if args.do_train:
        if TRAIN not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = tokenized_datasets[TRAIN]
        if args.max_train_samples is not None:
            train_dataset = train_dataset.shuffle(args.seed).select(range(args.max_train_samples))
    else:
        train_dataset = None

    if args.do_eval:
        if VALID not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        valid_dataset = tokenized_datasets[VALID]
        if args.max_eval_samples is not None:
            valid_dataset = valid_dataset.shuffle(args.seed).select(range(args.max_eval_samples))
    else:
        valid_dataset = None

    if args.do_predict:
        if TEST not in tokenized_datasets:
            raise ValueError("--do_predict requires a test dataset")
        test_dataset = tokenized_datasets[TEST]
        if args.max_predict_samples is not None:
            test_dataset = test_dataset.shuffle(args.seed).select(range(args.max_predict_samples))
    else:
        test_dataset = None
    
    return train_dataset, valid_dataset, test_dataset
