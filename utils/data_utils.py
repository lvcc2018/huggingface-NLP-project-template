from datasets import load_dataset
from consts import *


def load_dataset_from_files(args, suffix=JSON):
    data_dir = DATA_DIR / args.dataset
    data_files = {
        TRAIN: data_dir / args.train_file,
        VALID: data_dir / args.valid_file,
        TEST: data_dir / args.test_file
    }
    datasets = load_dataset(suffix, data_files=data_files)
    return datasets
