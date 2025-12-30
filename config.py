import panel as pn

# seed
seed = 42

# grad_cam hyperparameters
batch_size_gc = 32
learning_rate_gc = 0.001
epochs_gc = 5

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
    name="Random Perspective Distortion Scale", start=0.0, end=1.0, step=0.1, value=0.8
)
random_rotation_degrees_option = pn.widgets.IntSlider(
    name="Random Rotation Degrees", start=0, end=180, step=10, value=180
)
random_affine_degrees_option = pn.widgets.IntSlider(
    name="Random Affine Degrees", start=0, end=180, step=10, value=180
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
