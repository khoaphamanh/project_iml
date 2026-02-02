import torch
import os
import random
import numpy as np


class ConfigCGC:
    def __init__(self, seed=42):

        # seed
        self.seed = seed

        # model
        self.n = 2
        self.num_workers = 4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # training hyperparameters mvtec
        self.n_mvt = 30
        self.batch_size_mvt = 32
        self.learning_rate_mvt = 0.001
        self.epochs_mvt = 30
        self.consistency_mvt = True
        self.temperature_mvt = 0.5
        self.beta_mvt = 1
        self.n_channels_encoder_base_mvt = 16

        # path config
        self.path_config = os.path.abspath(__file__)
        self.path_mother_dir = os.path.dirname(self.path_config)

        # path dataset
        self.path_data_dir = os.path.join(self.path_mother_dir, "data")
        self.path_data_mvtec = os.path.join(self.path_data_dir, "MVTec_dataset")
        self.seafile_url = (
            "https://seafile.cloud.uni-hannover.de/f/d007fcef288648378676/?dl=1"
        )

        # dataset
        self.kind_data = ["train", "test"]

        # path mvt
        self.path_mvt_dir = os.path.join(self.path_mother_dir, "mvt")

        # path torch cache
        self.path_torch_cache = os.path.join(self.path_mvt_dir, "torch_cache")
        os.makedirs(self.path_torch_cache, exist_ok=True)
        torch.hub.set_dir(self.path_torch_cache)

        # path pretrained mvtec models
        self.path_pretrained_models_mvt_dir = os.path.join(
            self.path_mvt_dir, "pretrained_models"
        )
        os.makedirs(self.path_pretrained_models_mvt_dir, exist_ok=True)
        self.path_pretrained_models_mvt_pth = os.path.join(
            self.path_pretrained_models_mvt_dir,
            f"mvt_model_{self.seed}_{self.batch_size_mvt}_{self.learning_rate_mvt}_{self.epochs_mvt}_{self.consistency_mvt}_{int(self.learning_rate_mvt*100)}_{self.temperature_mvt}_{self.beta_mvt}.pth",
        )

        # path tko saved
        self.path_plot_results_dir = os.path.join(self.path_mother_dir, "plot_results")
        os.makedirs(self.path_plot_results_dir, exist_ok=True)
        self.path_visualize_dir = os.path.join(self.path_plot_results_dir, "visualize")
        os.makedirs(self.path_visualize_dir, exist_ok=True)

        self.path_tko_svg = os.path.join(self.path_visualize_dir, f"tko.svg")
        self.path_tko_png = os.path.join(self.path_visualize_dir, f"tko.png")

        self.path_accuracy_plot_svg = os.path.join(
            self.path_visualize_dir, f"accuracy_plot.svg"
        )
        self.path_accuracy_plot_png = os.path.join(
            self.path_visualize_dir, f"accuracy_plot.png"
        )

        self.path_win_rate_svg = os.path.join(self.path_visualize_dir, f"win_rate.svg")
        self.path_win_rate_png = os.path.join(self.path_visualize_dir, f"win_rate.png")

        self.path_loss_accuracy_svg = os.path.join(
            self.path_visualize_dir, f"loss_accuracy.svg"
        )
        self.path_loss_accuracy_png = os.path.join(
            self.path_visualize_dir, f"loss_accuracy.png"
        )

        # path stats and metrics
        self.path_results_dir = os.path.join(self.path_plot_results_dir, "results")
        os.makedirs(self.path_results_dir, exist_ok=True)
        self.path_results_pkl = os.path.join(
            self.path_results_dir, f"results_{self.seed}.pkl"
        )


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Critical for reproducibility on GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
