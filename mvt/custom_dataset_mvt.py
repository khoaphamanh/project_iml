import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path


# class MultiTaskDataset(Dataset):
#     def __init__(self, image_root, heatmap_root, transform=None):
#         self.image_root = Path(image_root)
#         self.heatmap_root = Path(heatmap_root)
#         self.transform = transform

#         # 1. Collect all image paths from the subdirectories
#         # This gets every file inside any subfolder of image_root
#         self.image_paths = sorted(list(self.image_root.glob("*/*.*")))

#         # 2. Extract class names from folder names to create a label map
#         self.classes = sorted([d.name for d in self.image_root.iterdir() if d.is_dir()])
#         self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         # --- 1. Path Logic ---
#         img_path = self.image_paths[idx]

#         # Get the relative path (e.g., 'class_A/img_001.jpg')
#         relative_path = img_path.relative_to(self.image_root)

#         # Construct heatmap path using the same subfolder/filename
#         hm_path = self.heatmap_root / relative_path

#         # --- 2. Loading Data ---
#         # Input image (X)
#         x = Image.open(img_path).convert("RGB")

#         # Classification label (Y_cls)
#         class_name = img_path.parent.name
#         y_cls = torch.tensor(self.class_to_idx[class_name], dtype=torch.long)

#         # Heatmap ground truth (Y_hm)
#         y_hm = Image.open(hm_path).convert("L")  # 'L' for 1-channel grayscale

#         # --- 3. Synchronized Transformations ---
#         if self.transform:
#             # We pass both x and y_hm to the transform to keep them aligned
#             x, y_hm = self.transform(x, y_hm)

#         return x, y_cls, y_hm

import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import numpy as np


class MultiTaskDataset(Dataset):
    def __init__(self, image_root, ground_truth_root, transform=None):
        self.image_root = Path(image_root)
        self.heatmap_root = Path(ground_truth_root)
        self.transform = transform

        # Collect all image paths (ignoring hidden files)
        self.image_paths = sorted(
            [f for f in self.image_root.glob("*/*.*") if not f.name.startswith(".")]
        )

        # Extract class names for labeling
        self.classes = sorted([d.name for d in self.image_root.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # path
        img_path = self.image_paths[idx]
        class_name = img_path.parent.name

        # Construct heatmap path with the _mask suffix
        # Example: images/class_A/001.png -> ground_truth/class_A/001_mask.png
        gt_filename = f"{img_path.stem}_mask{img_path.suffix}"
        gt_path = self.heatmap_root / class_name / gt_filename

        # Loading Data
        x = Image.open(img_path).convert("RGB")
        y_cls = torch.tensor(self.class_to_idx[class_name], dtype=torch.long)

        # Verify mask exists before opening to avoid crash
        if not gt_path.exists():
            raise FileNotFoundError(f"Missing mask file: {gt_path}")

        y_gt = Image.open(gt_path).convert("L")

        # mask_array = np.array(y_gt)
        # print(f"Unique values in mask: {np.unique(mask_array)}")

        # Synchronized Transformations
        if self.transform:
            x, y_gt = self.transform(x, y_gt)

        # only white (1) and black (0) pixels
        y_gt = (y_gt > 0.5).float()

        return x, y_gt, y_cls


if __name__ == "__main__":
    dataset = MultiTaskDataset(
        image_root="path/to/images",
        ground_truth_root="path/to/heatmaps",
        transform=None,
    )
    print(len(dataset))
