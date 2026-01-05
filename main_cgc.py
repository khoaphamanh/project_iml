from cgc.run_cgc import RunCGC
import config
import panel as pn
import random
import numpy as np
import torch
import os
from main import set_seed


def main():

    seed = config.seed
    batch_size_gc = config.batch_size_gc
    learning_rate_gc = config.learning_rate_gc
    epochs_gc = config.epochs_gc

    # set seed
    set_seed(seed)

    # run
    run_cgc = RunCGC(seed=seed)
    dashboard = run_cgc.train(
        batch_size=batch_size_gc,
        learning_rate=learning_rate_gc,
        epochs=epochs_gc,
        temperature=0.1,
        beta=0.5,
    )
    # dashboard.show()


if __name__ == "__main__":
    main()
