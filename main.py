from grad_cam.run_grad_cam import RunGradCAMImage
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

    print(f"Random seed set to {seed}")


def main():

    seed = config.seed
    batch_size_gc = config.batch_size_gc
    learning_rate_gc = config.learning_rate_gc
    epochs_gc = config.epochs_gc

    # set seed
    set_seed(seed)

    # augmentation options
    random_crop_size_option = config.random_crop_size_option
    color_jitter_brightness_option = config.color_jitter_brightness_option
    color_jitter_contrast_option = config.color_jitter_contrast_option
    color_jitter_saturation_option = config.color_jitter_saturation_option
    color_jitter_hue_option = config.color_jitter_hue_option
    random_perspective_distortion_scale_option = (
        config.random_perspective_distortion_scale_option
    )
    random_rotation_degrees_option = config.random_rotation_degrees_option
    random_affine_degrees_option = config.random_affine_degrees_option
    random_affine_translate_option = config.random_affine_translate_option
    random_affine_scale_min_option = config.random_affine_scale_min_option
    random_affine_scale_max_option = config.random_affine_scale_max_option
    random_affine_shear_option = config.random_affine_shear_option

    # run
    run_grad_cam = RunGradCAMImage(seed=seed)
    run_grad_cam.run_image_grad_cam(
        batch_size=batch_size_gc,
        learning_rate=learning_rate_gc,
        epochs=epochs_gc,
        random_crop_size_option=random_crop_size_option,
        color_jitter_brightness_option=color_jitter_brightness_option,
        color_jitter_contrast_option=color_jitter_contrast_option,
        color_jitter_saturation_option=color_jitter_saturation_option,
        color_jitter_hue_option=color_jitter_hue_option,
        random_perspective_distortion_scale_option=random_perspective_distortion_scale_option,
        random_rotation_degrees_option=random_rotation_degrees_option,
        random_affine_degrees_option=random_affine_degrees_option,
        random_affine_translate_option=random_affine_translate_option,
        random_affine_scale_min_option=random_affine_scale_min_option,
        random_affine_scale_max_option=random_affine_scale_max_option,
        random_affine_shear_option=random_affine_shear_option,
    )


if __name__ == "__main__":
    main()
