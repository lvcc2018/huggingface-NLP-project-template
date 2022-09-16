from torch.utils.data import DataLoader
from dataset import CustomDataset
from src.consts import *


def load_dataset_from_files(args):
    data_dir = DATA_DIR / args.dataset
    data_files = {
        TRAIN: data_dir / args.train_file,
        VALID: data_dir / args.valid_file,
        TEST: data_dir / args.test_file
    }
    raw_datasets = {s: CustomDataset(args, data_files[s], s) for s in SPLITS}
    return raw_datasets


def preprocess_datasets(args, raw_datasets, tokenizer, logger):

    # Padding strategy
    if args.pad_to_max_length:
        padding = "max_length"
    else:
        padding = False

    if args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(instance):
        # Process the data here
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
            train_dataset = train_dataset.shuffle(
                args.seed).select(range(args.max_train_samples))
    else:
        train_dataset = None

    if args.do_eval:
        if VALID not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        valid_dataset = tokenized_datasets[VALID]
        if args.max_eval_samples is not None:
            valid_dataset = valid_dataset.shuffle(
                args.seed).select(range(args.max_eval_samples))
    else:
        valid_dataset = None

    if args.do_predict:
        if TEST not in tokenized_datasets:
            raise ValueError("--do_predict requires a test dataset")
        test_dataset = tokenized_datasets[TEST]
        if args.max_predict_samples is not None:
            test_dataset = test_dataset.select(range(args.max_predict_samples))
    else:
        test_dataset = None

    return train_dataset, valid_dataset, test_dataset


def prepare_dataloader(args, tokenized_dataset, tokenizer):
    if not tokenized_dataset:
        return None

    def collate_fn(instances):
        tokenizer.pad(instances)
        return instances
    dataloader = DataLoader(
        tokenized_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    return dataloader


def get_dataloaders(args, train_dataset, valid_dataset, test_dataset, tokenizer):
    train_dataloader = prepare_dataloader(args, train_dataset, tokenizer)
    valid_dataloader = prepare_dataloader(args, valid_dataset, tokenizer)
    test_dataloader = prepare_dataloader(args, test_dataset, tokenizer)
    return train_dataloader, valid_dataloader, test_dataloader
