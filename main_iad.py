from iad.run_iad import RunIAD
import config
import panel as pn
import random
import numpy as np
import torch
import os


def set_seed(seed=42):

    # Python's built-in random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # PyTorch backends
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Python hash seed (for Python 3.3+)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():

    seed = config.seed
    batch_size_iad = 32
    learning_rate_iad = 0.001
    epochs_iad = 10

    # set seed
    set_seed(seed)

    # run
    run_iad = RunIAD(seed=seed)
    dataset_train, dataset_test = run_iad.load_data_image()
    dataset_train_classes = dataset_train.class_to_idx
    print("dataset_train_classes:", dataset_train_classes)
    dataset_test_classes = dataset_test.class_to_idx
    print("dataset_test_classes:", dataset_test_classes)


if __name__ == "__main__":
    main()
