import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchinfo import summary
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
import os
from torchvision.transforms import InterpolationMode
from .model_custom import EfficientNetB0Modified
import random
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from torchinfo import summary
from .grad_cam import get_gradcam
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import panel as pn
import matplotlib.cm as cm

pn.extension("plotly")


class RunGradCAMImage:
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
            self.path_data_directory, "dataset_image"
        )
        self.name_classes = os.listdir(self.path_dataset_image_directory + "/train")
        self.kind_data = ["train", "test"]

        # path pretrained model
        self.path_initialize_model_cache = os.path.join(
            self.path_grad_cam_dir, "torch_cache"
        )
        torch.hub.set_dir(self.path_initialize_model_cache)

        # path grad_cam
        self.path_grad_cam_pretrained_model_directory = os.path.join(
            self.path_grad_cam_dir, "pretrained_model"
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

    def load_class_weighted_cross_entropy_loss(
        self, dataset_train: datasets.ImageFolder
    ):

        # get labels
        y = np.array([dataset_train[i][1] for i in range(len(dataset_train))]).flatten()

        # Identify the unique classes in the target labels
        classes = np.unique(y)

        # Compute class weights to handle class imbalance
        class_weights = compute_class_weight("balanced", classes=classes, y=y)

        # Convert class weights to a tensor and move to the appropriate device
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(self.device)

        # Initialize the CrossEntropyLoss with the computed class weights
        loss = nn.CrossEntropyLoss(weight=class_weights)

        return loss

    def iteration_loop(
        self,
        model: EfficientNetB0Modified,
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
        loss_total = 0

        # initial saved array
        array_y_true = []
        array_y_pred = []

        # iteration loop
        with mode:
            for iter_mode, (x, y) in enumerate(dataloader):

                # move to device
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                # forward pass
                y_logits = model(x)
                y_pred = torch.argmax(y_logits, dim=1)

                # compute loss
                loss_this_batch = loss(y_logits, y)
                loss_total += loss_this_batch.item()

                # calculate gradient and update weights
                if enable_grad:
                    optimizer.zero_grad()
                    loss_this_batch.backward()
                    optimizer.step()

                # save prediction and true label
                array_y_true.extend(y.detach().cpu().numpy())
                array_y_pred.extend(y_pred.detach().cpu().numpy())

        # calculate mean loss
        loss_total = loss_total / len(dataloader)

        # calculate accuracy and confusion matrix
        acc_bal = balanced_accuracy_score(array_y_true, array_y_pred)
        cm = confusion_matrix(array_y_true, array_y_pred)

        if print_out:
            print(
                f"Epoch: {ep}, {kind_data} loss: {loss_total:.4f}, balanced accuracy: {acc_bal:.4f}"
            )
            print("Confusion matrix:")
            print(cm)

    def create_nine_augmentations(
        self,
        image,
        random_crop_size,
        color_jitter_brightness,
        color_jitter_contrast,
        color_jitter_saturation,
        color_jitter_hue,
        random_perspective_distortion_scale,
        random_rotation_degrees,
        random_affine_degrees,
        random_affine_translate,
        random_affine_scale_min,
        random_affine_scale_max,
        random_affine_shear,
    ):
        # list image and name
        list_images = []
        names = []

        # list augmentations
        list_augmentations = [
            ("Original", transforms.Compose([])),
            (
                "RandomCrop",
                transforms.Compose(
                    [
                        transforms.RandomCrop(
                            size=(random_crop_size, random_crop_size)
                        ),
                        transforms.Resize(224),
                    ]
                ),
            ),
            ("RandomInvert", transforms.RandomInvert(p=1.0)),
            ("Grayscale", transforms.Grayscale(num_output_channels=3)),
            (
                "ColorJitter",
                transforms.ColorJitter(
                    brightness=color_jitter_brightness,
                    contrast=color_jitter_contrast,
                    saturation=color_jitter_saturation,
                    hue=color_jitter_hue,
                ),
            ),
            (
                "RandomPerspective",
                transforms.RandomPerspective(
                    distortion_scale=random_perspective_distortion_scale, p=1.0
                ),
            ),
            (
                "RandomRotation",
                transforms.RandomRotation(degrees=random_rotation_degrees),
            ),
            (
                "RandomAffine",
                transforms.RandomAffine(
                    degrees=random_affine_degrees,
                    translate=(random_affine_translate, random_affine_translate),
                    scale=(random_affine_scale_min, random_affine_scale_max),
                    shear=random_affine_shear,
                ),
            ),
            ("HorizontalFlip", transforms.RandomHorizontalFlip(p=1.0)),
            ("VerticalFlip", transforms.RandomVerticalFlip(p=1.0)),
        ]

        for name, aug in list_augmentations:
            augmented_image = aug(image)
            list_images.append(augmented_image)
            names.append(name)

        images = torch.stack(list_images, dim=0)
        return images, names

    def train(
        self, batch_size: int = 32, learning_rate: float = 0.001, epochs: int = 50
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
        loss = self.load_class_weighted_cross_entropy_loss(dataset_train)
        optmizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # train model
        for ep in range(epochs):

            # training loop
            self.iteration_loop(
                model=model,
                dataloader=dataloader_train,
                optimizer=optmizer,
                loss=loss,
                ep=ep,
                enable_grad=True,
                kind_data="train",
            )

            # testing loop
            self.iteration_loop(
                model=model,
                dataloader=dataloader_test,
                optimizer=optmizer,
                loss=loss,
                ep=ep,
                enable_grad=False,
                kind_data="test",
            )
            print()

        # saved model
        dict_saved = {
            "model_state_dict": model.state_dict(),
            "dataset_train": dataset_train,
            "dataset_test": dataset_test,
        }

        path_model = os.path.join(
            self.path_grad_cam_pretrained_model_directory,
            f"model_image_{self.seed}_{epochs}_{batch_size}_{int(learning_rate*1000)}.pt",
        )
        torch.save(dict_saved, path_model)

    def plotly_grad_cam_on_image_and_augmentation(
        self,
        target_class_str,
        input_tensor,
        classes_dict_idx_to_str,
        grad_cam_original,
        grad_cam_upsampled,
        y_pred,
        y_prob,
        label,
        names_augmentation,
    ):

        # plot using plotly
        num_images = len(grad_cam_original)
        title_row_1 = [
            f"True: {classes_dict_idx_to_str[label]}<br>Pred: {classes_dict_idx_to_str[int(y_pred[i])]}<br>Prob: {y_prob[i]:.4f}<br>Aug: {names_augmentation[i]}"
            for i in range(num_images)
        ]
        title_row_2 = [
            f"Grad-CAM (Original)<br>shape: {grad_cam_original[0].shape}"
        ] + ["" for _ in range(1, num_images)]
        title_row_3 = [
            f"Grad-CAM (Upsampled)<br>shape: {grad_cam_upsampled[0].shape}"
        ] + ["" for _ in range(1, num_images)]
        title_row_4 = [
            f"Overlay (Image + Grad-CAM)<br>shape: {grad_cam_upsampled[0].shape}"
        ] + ["" for _ in range(1, num_images)]

        title = [*title_row_1, *title_row_2, *title_row_3, *title_row_4]

        fig = make_subplots(
            rows=4,
            cols=num_images,
            vertical_spacing=0.12,
            horizontal_spacing=0.02,
            subplot_titles=title,
        )

        # add images to plotly figure
        for i in range(num_images):
            # Row 1: Original images (convert tensor to RGB numpy array)
            # Permute from [C, H, W] to [H, W, C] and clamp to [0, 1]
            img_rgb = input_tensor[i].permute(1, 2, 0).cpu().numpy()
            img_rgb = (img_rgb - img_rgb.min()) / (
                img_rgb.max() - img_rgb.min() + 1e-8
            )  # Normalize to [0,1]

            fig.add_trace(
                go.Image(
                    z=(img_rgb * 255).astype("uint8"),  # Scale to 0-255 for display
                    hovertemplate=(
                        f"<b>Aug: {names_augmentation[i]}</b><br>"
                        f"True: {classes_dict_idx_to_str[label]}<br>"
                        f"Pred: {classes_dict_idx_to_str[int(y_pred[i])]}<br>"
                        f"Prob: {y_prob[i]:.2%}<br>"
                        "Pixel (x=%{x}, y=%{y})<extra></extra>"
                    ),
                ),
                row=1,
                col=i + 1,
            )

            # Row 2: Original Grad-CAM
            fig.add_trace(
                go.Heatmap(
                    z=grad_cam_original[i],
                    colorscale="jet",
                    zmin=0,
                    zmax=1,
                    zauto=False,
                    showscale=False,
                    hovertemplate=(
                        f"<b>Grad-CAM Original</b><br>"
                        f"Aug: {names_augmentation[i]}<br>"
                        "Pixel (x, y): (%{x}, %{y})<br>"
                        "Activation: %{z:.4f}<extra></extra>"
                    ),
                ),
                row=2,
                col=i + 1,
            )

            # Row 3: Upsampled Grad-CAM
            fig.add_trace(
                go.Heatmap(
                    z=grad_cam_upsampled[i],
                    colorscale="jet",
                    zmin=0,
                    zmax=1,
                    zauto=False,
                    showscale=False,
                    hovertemplate=(
                        f"<b>Grad-CAM Upsampled</b><br>"
                        f"Aug: {names_augmentation[i]}<br>"
                        "Pixel (x, y): (%{x}, %{y})<br>"
                        "Activation: %{z:.4f}<extra></extra>"
                    ),
                ),
                row=3,
                col=i + 1,
            )

            # Row 4: Overlay (Image + Grad-CAM)
            # Convert grad-cam to RGB heatmap
            colormap = cm.get_cmap("jet")
            heatmap_rgb = colormap(grad_cam_upsampled[i])[
                :, :, :3
            ]  # Remove alpha channel

            # Blend: 60% original image + 40% heatmap
            overlay = 0.6 * img_rgb + 0.4 * heatmap_rgb
            overlay = np.clip(overlay, 0, 1)  # Ensure values in [0, 1]

            fig.add_trace(
                go.Image(
                    z=(overlay * 255).astype("uint8"),
                    hovertemplate=(
                        f"<b>Overlay</b><br>"
                        f"Aug: {names_augmentation[i]}<br>"
                        f"True: {classes_dict_idx_to_str[label]}<br>"
                        f"Pred: {classes_dict_idx_to_str[int(y_pred[i])]}<br>"
                        "Pixel (x=%{x}, y=%{y})<extra></extra>"
                    ),
                ),
                row=4,
                col=i + 1,
            )

        # layout
        if target_class_str == "Predicted":
            target_class_str = (
                target_class_str + f" {classes_dict_idx_to_str[int(y_pred[0])]}"
            )
        elif target_class_str == "True":
            target_class_str = target_class_str + f" {classes_dict_idx_to_str[label]}"

        fig.update_layout(
            height=1200,
            width=2000,
            showlegend=False,
            title_text=f"Normalized Grad-CAM Batch Visualization, target class: {target_class_str}",
            margin=dict(t=180),
        )

        # Remove axes
        fig.update_xaxes(
            showticklabels=False,
            showgrid=False,
        )
        fig.update_yaxes(showticklabels=False, showgrid=False, autorange="reversed")

        return fig

    def image_grad_cam(
        self,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        epochs: int = 50,
        index: int = 0,
        kind_data: str = "train",
        target_class: str = "Predicted",
        random_crop_size: int = 100,
        color_jitter_brightness: float = 0.5,
        color_jitter_contrast: float = 0.5,
        color_jitter_saturation: float = 0.3,
        color_jitter_hue: float = 0.3,
        random_perspective_distortion_scale: float = 0.3,
        random_rotation_degrees: int = 10,
        random_affine_degrees: int = 10,
        random_affine_translate: int = 10,
        random_affine_scale_min: float = 0.8,
        random_affine_scale_max: float = 1.2,
        random_affine_shear: float = 10,
    ):

        # train model
        path_model = os.path.join(
            self.path_grad_cam_pretrained_model_directory,
            f"model_image_{self.seed}_{epochs}_{batch_size}_{int(learning_rate * 1000)}.pt",
        )
        if not os.path.exists(path_model):
            self.train(batch_size, learning_rate, epochs)

        # load pretrained model
        dict_saved = torch.load(path_model, weights_only=False)

        # load data
        dataset_train = dict_saved["dataset_train"]
        dataset_test = dict_saved["dataset_test"]
        model_state_dict = dict_saved["model_state_dict"]
        classes_dict_str_to_idx = dataset_train.class_to_idx
        classes_idx_to_str = {v: k for k, v in classes_dict_str_to_idx.items()}

        # load model
        num_classes = len(dataset_train.classes)
        model = EfficientNetB0Modified(num_classes=num_classes)
        model.load_state_dict(model_state_dict)
        model.to(self.device)
        model.eval()

        # get the img
        data = dataset_train if kind_data == "train" else dataset_test
        if index >= len(data):
            index = len(data) - 1
        img = Subset(data, [index])
        input_tensor, label = img[0]

        input_tensor, names_augmentation = self.create_nine_augmentations(
            input_tensor,
            random_crop_size,
            color_jitter_brightness,
            color_jitter_contrast,
            color_jitter_saturation,
            color_jitter_hue,
            random_perspective_distortion_scale,
            random_rotation_degrees,
            random_affine_degrees,
            random_affine_translate,
            random_affine_scale_min,
            random_affine_scale_max,
            random_affine_shear,
        )

        # target class
        if target_class == "Predicted":
            target_class_int = None
        elif target_class == "True":
            target_class_int = label
        else:
            target_class_int = classes_dict_str_to_idx[target_class]

        # get gradcam
        y_pred, y_prob, gradcam_original, gradcam_upsampled = get_gradcam(
            model=model,
            input_tensor=input_tensor,
            target_class_int=target_class_int,
        )

        # plot grad cam
        fig = self.plotly_grad_cam_on_image_and_augmentation(
            input_tensor=input_tensor,
            target_class_str=target_class,
            classes_dict_idx_to_str=classes_idx_to_str,
            grad_cam_original=gradcam_original,
            grad_cam_upsampled=gradcam_upsampled,
            y_pred=y_pred,
            y_prob=y_prob,
            label=label,
            names_augmentation=names_augmentation,
        )

        return fig

    def run_image_grad_cam(
        self,
        batch_size,
        learning_rate,
        epochs,
    ):
        # data options
        kind_data_option = pn.widgets.Select(
            name="Kind of Data", options=self.kind_data, value="train"
        )
        index_option = pn.widgets.IntSlider(
            name="Image Index", start=0, end=1500, step=1, value=0
        )
        # target class option
        target_class_option = pn.widgets.Select(
            name="Target Class",
            options=self.name_classes + ["Predicted", "True"],
            value="Predicted",
        )

        # grad_cam augmentation options
        random_crop_size_option = pn.widgets.IntSlider(
            name="Random Crop Size", start=50, end=200, step=10, value=100
        )
        color_jitter_brightness_option = pn.widgets.FloatSlider(
            name="Color Jitter Brightness", start=0.0, end=1.0, step=0.1, value=0.5
        )
        color_jitter_contrast_option = pn.widgets.FloatSlider(
            name="Color Jitter Contrast", start=0.0, end=1.0, step=0.1, value=0.5
        )
        color_jitter_saturation_option = pn.widgets.FloatSlider(
            name="Color Jitter Saturation", start=0.0, end=1.0, step=0.1, value=0.5
        )
        color_jitter_hue_option = pn.widgets.FloatSlider(
            name="Color Jitter Hue", start=0.0, end=0.5, step=0.1, value=0.3
        )
        random_perspective_distortion_scale_option = pn.widgets.FloatSlider(
            name="Random Perspective Distortion Scale",
            start=0.0,
            end=1.0,
            step=0.1,
            value=0.8,
        )
        random_rotation_degrees_option = pn.widgets.IntSlider(
            name="Random Rotation Degrees", start=0, end=360, step=10, value=180
        )
        random_affine_degrees_option = pn.widgets.IntSlider(
            name="Random Affine Degrees", start=0, end=360, step=10, value=180
        )
        random_affine_translate_option = pn.widgets.FloatSlider(
            name="Random Affine Translate", start=0.0, end=0.5, step=0.05, value=0.5
        )
        random_affine_scale_min_option = pn.widgets.FloatSlider(
            name="Random Affine Scale Min", start=0.5, end=1.0, step=0.05, value=0.9
        )
        random_affine_scale_max_option = pn.widgets.FloatSlider(
            name="Random Affine Scale Max", start=1.0, end=1.5, step=0.05, value=1.5
        )
        random_affine_shear_option = pn.widgets.IntSlider(
            name="Random Affine Shear", start=0, end=45, step=5, value=10
        )

        # bind interaction
        interactive_gradcam = pn.bind(
            self.image_grad_cam,
            # hyperparameters
            batch_size=batch_size,
            learning_rate=learning_rate,
            epochs=epochs,
            # data options
            kind_data=kind_data_option,
            index=index_option,
            target_class=target_class_option,
            # augmentation options
            random_crop_size=random_crop_size_option,
            color_jitter_brightness=color_jitter_brightness_option,
            color_jitter_contrast=color_jitter_contrast_option,
            color_jitter_saturation=color_jitter_saturation_option,
            color_jitter_hue=color_jitter_hue_option,
            random_perspective_distortion_scale=random_perspective_distortion_scale_option,
            random_rotation_degrees=random_rotation_degrees_option,
            random_affine_degrees=random_affine_degrees_option,
            random_affine_translate=random_affine_translate_option,
            random_affine_scale_min=random_affine_scale_min_option,
            random_affine_scale_max=random_affine_scale_max_option,
            random_affine_shear=random_affine_shear_option,
        )

        # dashboard layout
        dashboard = pn.Column(
            "# Grad-CAM Visualization Dashboard",
            pn.Row(
                pn.Column(
                    random_crop_size_option,
                    color_jitter_brightness_option,
                    color_jitter_contrast_option,
                    color_jitter_saturation_option,
                    color_jitter_hue_option,
                    random_perspective_distortion_scale_option,
                    random_rotation_degrees_option,
                    random_affine_degrees_option,
                    random_affine_translate_option,
                    random_affine_scale_min_option,
                    random_affine_scale_max_option,
                    random_affine_shear_option,
                ),
                pn.Column(
                    kind_data_option,
                    index_option,
                    target_class_option,
                ),
            ),
            interactive_gradcam,
        )

        return dashboard
