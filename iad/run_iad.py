import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchinfo import summary
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
import os
from torchvision.transforms import InterpolationMode
from .model_custom import EfficientNetB0Autoencoder
import random
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from torchinfo import summary

# from .grad_cam import get_gradcam
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import panel as pn
import matplotlib.cm as cm

pn.extension("plotly")


class RunIAD:

    def __init__(self, seed: int = 42):

        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_workers = 4

        # seed
        self.seed = seed

        # path of script and grad_cam directory
        self.path_script = os.path.abspath(__file__)
        self.path_grad_cam_dir = os.path.dirname(self.path_script)

        # path mother directory
        self.path_mother_directory = os.path.dirname(self.path_grad_cam_dir)

        # path data and dataset directory
        self.path_data_directory = os.path.join(self.path_mother_directory, "data")
        self.path_dataset_image_directory = os.path.join(
            self.path_data_directory, "dataset_mvtec"
        )
        self.name_classes = [
            i
            for i in os.listdir(self.path_dataset_image_directory + "/train")
            if "." not in i
        ]
        self.kind_data = ["train", "test"]

        # path pretrained model
        self.path_initialize_model_cache = os.path.join(
            self.path_grad_cam_dir, "torch_cache"
        )
        torch.hub.set_dir(self.path_initialize_model_cache)

        # path grad_cam
        self.path_grad_cam_pretrained_model_directory = os.path.join(
            self.path_grad_cam_dir, "pretrained_model_iad"
        )
        os.makedirs(self.path_grad_cam_pretrained_model_directory, exist_ok=True)

    def seed_worker(self, worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def load_data_image(self):

        # resize to 256 and then center crop for transform
        transform_image = transforms.Compose(
            [
                transforms.Resize(
                    size=256, interpolation=InterpolationMode.BICUBIC, antialias=True
                ),
                transforms.ToTensor(),
                transforms.CenterCrop(size=224),
            ]
        )

        # load dataset
        dataset_train = datasets.ImageFolder(
            root=self.path_dataset_image_directory + "/train",
            transform=transform_image,
        )

        dataset_test = datasets.ImageFolder(
            root=self.path_dataset_image_directory + "/test", transform=transform_image
        )

        return dataset_train, dataset_test

    def iteration_loop(
        self,
        model: EfficientNetB0Autoencoder,
        dataloader: DataLoader,
        optimizer: torch.optim,
        loss: nn.CrossEntropyLoss,
        ep: int,
        enable_grad: bool = True,
        kind_data: str = "train",
        print_out: bool = True,
    ):

        # Determine whether to enable gradient computation
        if enable_grad:
            mode = torch.enable_grad()
            model.train()
        else:
            mode = torch.no_grad()
            model.eval()

        # Initialize loss total
        list_loss_total = []

        # iteration loop
        with mode:
            for iter_mode, (x, y) in enumerate(dataloader):

                # move to device
                x = x.to(self.device, non_blocking=True)

                # forward pass
                x_ = model(x)

                # loss
                loss_this_batch = loss(x_, x)
                list_loss_total.extend(loss_this_batch.detach().cpu().numpy())
                loss_this_batch = loss_this_batch.mean()

                # calculate gradient and update weights
                if enable_grad:
                    optimizer.zero_grad()
                    loss_this_batch.backward()
                    optimizer.step()

        # calculate mean loss
        loss_total = np.mean(list_loss_total)

        if print_out:
            print(f"Epoch: {ep}{kind_data}, Loss: {loss_total:.4f}")

    def train(
        self, batch_size: int = 32, learning_rate: float = 0.001, epochs: int = 10
    ):

        # load dataset
        dataset_train, dataset_test = self.load_data_image()

        # dataloader
        dataloader_train = DataLoader(
            dataset=dataset_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            worker_init_fn=self.seed_worker,
            pin_memory=True,
            persistent_workers=True,
            generator=torch.Generator().manual_seed(self.seed),
        )
        dataloader_test = DataLoader(
            dataset=dataset_test,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            worker_init_fn=self.seed_worker,
            pin_memory=True,
            persistent_workers=True,
            generator=torch.Generator().manual_seed(self.seed),
        )

        # load model
        num_classes = len(dataset_train.classes)
        model = EfficientNetB0Autoencoder(out_channels=3, base_channels=128)
        model.to(self.device)

        # optimizer
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # loss
        loss = nn.MSELoss(reduction="none")

        # epoch loop
        for ep in range(epochs):
            self.iteration_loop(
                model=model,
                dataloader=dataloader_train,
                optimizer=optimizer,
                loss=loss,
                ep=ep,
                enable_grad=True,
                kind_data="train",
            )

            self.iteration_loop(
                model=model,
                dataloader=dataloader_test,
                optimizer=optimizer,
                loss=loss,
                ep=ep,
                enable_grad=False,
                kind_data="test",
            )

        # save model
        torch.save(
            model.state_dict(),
            os.path.join(self.path_grad_cam_pretrained_model_directory, "model.pth"),
        )
