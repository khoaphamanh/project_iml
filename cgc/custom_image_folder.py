from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.transforms import functional as F
import random


class ImageFolderCustom(ImageFolder):
    def __init__(
        self,
        root,
        transform=None,
        target_transform=None,
        loader=...,
        is_valid_file=None,
        allow_empty=False,
    ):
        super().__init__(
            root,
            transform,
            target_transform,
            is_valid_file=is_valid_file,
            allow_empty=allow_empty,
        )

        # transform image for grad cam
        self.resize = transforms.Resize(256)
        self.center_crop = transforms.CenterCrop(224)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        # horizontal flip
        self.hor_flip = transforms.RandomHorizontalFlip(p=1.0)

    def __getitem__(self, index):

        # get original image
        path, target = self.samples[index]
        img = self.loader(path)

        # image for cross entropy loss
        x_ce = self.transform(img)

        # image for grad cam original (first only resize)
        x_grad_cam_original = self.resize(img)
        x_grad_cam_original = self.center_crop(x_grad_cam_original)

        # image for grad cam aug random resize crop
        i, j, h, w = transforms.RandomResizedCrop.get_params(
            x_grad_cam_original, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333)
        )
        x_grad_cam_aug = F.resized_crop(
            x_grad_cam_original, i, j, h, w, size=(224, 224)
        )

        # horizontal flip
        p = random.random()
        hor_flip = False
        if p > 0.5:
            x_grad_cam_aug = self.hor_flip(x_grad_cam_aug)
            hor_flip = True

        # image for grad cam aug to tensor and normalize
        x_grad_cam_aug = self.to_tensor(x_grad_cam_aug)
        x_grad_cam_aug = self.normalize(x_grad_cam_aug)

        # image for grad cam original to tensor and normalize
        x_grad_cam_original = self.to_tensor(x_grad_cam_original)
        x_grad_cam_original = self.normalize(x_grad_cam_original)

        return x_ce, x_grad_cam_original, x_grad_cam_aug, i, j, h, w, hor_flip, target


if __name__ == "__main__":

    import torch

    base_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    transform_grad_cam = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    path = "/Volumes/khoa_ssd/uni/iml_2025/project_iml/data/dataset_image/train"

    dataset_example = ImageFolderCustom(root=path, transform=base_transform)

    instance = dataset_example[0]
    for i in instance:
        if isinstance(i, torch.Tensor):
            print(i.shape)
        else:
            print(i)


def transform_grad_cam_given_parameters(grad_cam_ori, i, j, h, w, hor_flip):

    batch_size = grad_cam_ori.shape[0]
    grad_cam_transformed = []

    for idx in range(batch_size):

        # apply crop
        gc_crop = F.resized_crop(
            grad_cam_ori[idx],
            i[idx].item(),
            j[idx].item(),
            h[idx].item(),
            w[idx].item(),
            size=(224, 224),
        )

        # apply horizontal flip
        if hor_flip[idx]:
            gc_crop = F.hflip(gc_crop)

        grad_cam_transformed.append(gc_crop)

    return torch.stack(grad_cam_transformed)
