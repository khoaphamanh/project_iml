import torch
import os
import numpy as np
import random
from .custom_image_folder import ImageFolderCustom
from .simclr_loss import ContrastiveNTXentLoss
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.utils.class_weight import compute_class_weight
from .model_custom import EfficientNetB0Modified
from torchinfo import summary
from .grad_cam_cgc import get_gradcam


class RunCGC:
    def __init__(self, seed=42):

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
            self.path_data_directory, "dataset_image"
        )
        self.name_classes = [
            i
            for i in os.listdir(self.path_dataset_image_directory + "/train")
            if "." not in i
        ]
        self.kind_data = ["train", "test"]

        # path pretrained model
        self.path_initialize_model_cache = os.path.join(
            self.path_grad_cam_dir, "torch_cache_cgc"
        )
        torch.hub.set_dir(self.path_initialize_model_cache)

        # path grad_cam
        self.path_grad_cam_pretrained_model_directory = os.path.join(
            self.path_grad_cam_dir, "pretrained_model_cgc"
        )
        os.makedirs(self.path_grad_cam_pretrained_model_directory, exist_ok=True)

    def seed_worker(self, worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def load_class_weighted_cross_entropy_loss(self, dataset_train: ImageFolderCustom):

        # get labels
        y = np.array(
            [dataset_train[i][-1] for i in range(len(dataset_train))]
        ).flatten()

        # Identify the unique classes in the target labels
        classes = np.unique(y)

        # Compute class weights to handle class imbalance
        class_weights = compute_class_weight("balanced", classes=classes, y=y)

        # Convert class weights to a tensor and move to the appropriate device
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(self.device)

        # Initialize the CrossEntropyLoss with the computed class weights
        loss = nn.CrossEntropyLoss(weight=class_weights)

        return loss

    def load_data_image_cgc(self):
        transform_base = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        dataset_train = ImageFolderCustom(
            root=self.path_dataset_image_directory + "/train", transform=transform_base
        )

        dataset_test = ImageFolderCustom(
            root=self.path_dataset_image_directory + "/test", transform=transform_base
        )

        return dataset_train, dataset_test

    def iteration_loop_cgc(
        self,
        model: EfficientNetB0Modified,
        dataloader: DataLoader,
        optimizer: torch.optim,
        loss: nn.CrossEntropyLoss,
        ep: int,
        beta: float = 0.5,
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
        loss_total = 0

        # initial saved array
        array_y_true = []
        array_y_pred = []

        # iteration loop
        with mode:
            for iter_mode, (
                x_ce,
                x_gc_ori,
                x_gc_aug,
                i,
                j,
                h,
                w,
                hor_flip,
                y,
            ) in enumerate(dataloader):

                # print all the shape
                print(
                    f"x_ce: {x_ce.shape}, x_gc_ori: {x_gc_ori.shape}, x_gc_aug: {x_gc_aug.shape}, i: {i.shape}, j: {j.shape}, h: {h.shape}, w: {w.shape}, hor_flip: {hor_flip.shape}, y: {y.shape}"
                )

                # forward pass ce
                x = x_ce.to(self.device)
                y_ce_logits = model(x)

                # forward pass sample gradcam
                x = x_gc_ori.to(self.device)
                feature_map_gc_ori, y_gc_ori_logits = model.get_feature_maps(x)
                print("feature_map_gc_ori shape:", feature_map_gc_ori.shape)
                target_class_gc = torch.argmax(y_gc_ori_logits, dim=1)

                # forward pass aug gradcam
                x = x_gc_aug.to(self.device)
                feature_map_gc_aug, y_gc_aug_logits = model.get_feature_maps(x)
                print("feature_map_gc_aug shape:", feature_map_gc_aug.shape)

                # calculate grad cam
                _, _, _, grad_cam_gc = get_gradcam(
                    model, feature_map_gc_ori, target_class_int=target_class_gc
                )
                _, _, _, grad_cam_gc_aug = get_gradcam(
                    model, feature_map_gc_aug, target_class_int=target_class_gc
                )

                # transform grad cam ori with given parameters

    def train(
        self,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        epochs: int = 50,
        temperature: float = 0.1,
        beta: float = 0.5,
    ):

        # load dataset
        dataset_train, dataset_test = self.load_data_image_cgc()

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
        model = EfficientNetB0Modified(num_classes=num_classes)
        model.to(self.device)

        # summary the model
        summary(
            model,
            input_size=(batch_size, 3, 224, 224),
            col_names=(
                "input_size",
                "output_size",
                "num_params",
                "mult_adds",
                "trainable",
            ),
            verbose=1,
        )

        # loss and optimizer
        loss_ce = self.load_class_weighted_cross_entropy_loss(dataset_train)
        optmizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss_simclr = ContrastiveNTXentLoss(temperature=temperature)

        # train model
        for ep in range(epochs):

            self.iteration_loop_cgc(
                model=model,
                dataloader=dataloader_train,
                optimizer=optmizer,
                loss=loss_ce,
                ep=ep,
                beta=beta,
                enable_grad=True,
                kind_data="train",
                print_out=True,
            )
