from .model_custom import EfficientNetB0Modified
import torch
import torch.nn.functional as F


def normalize_batch_gradcam(tensor, batch_size):
    """Normalize each sample in batch to [0, 1] independently."""
    flat = tensor.view(batch_size, -1)
    min_vals = flat.min(dim=1, keepdim=True)[0].view(batch_size, 1, 1)
    max_vals = flat.max(dim=1, keepdim=True)[0].view(batch_size, 1, 1)
    return (tensor - min_vals) / (max_vals - min_vals + 1e-8)


def get_gradcam(
    model: EfficientNetB0Modified,
    feature_map: torch.Tensor,
    logits: torch.Tensor,
    target_class_int=None,
):

    # model in eval mode
    model.eval()
    batch_size = feature_map.shape[0]

    # unfreeze model
    model.unfreeze_features()

    feature_map.retain_grad()

    if target_class_int is None:
        # target class as predicted class
        target_class_int = logits.argmax(dim=1)

    else:
        if isinstance(target_class_int, int):
            # given target class as string
            target_class_int = torch.tensor([target_class_int] * batch_size)

        else:
            # target class as list of y_true labels
            target_class_int = torch.tensor(target_class_int)

    # y_pred
    y_pred = logits.argmax(dim=1)
    y_prob = torch.softmax(logits, dim=1).amax(dim=1).detach().numpy()

    # zero gradients
    model.zero_grad()

    # Create one-hot encoding for target classes: sum over batch for efficiency
    print("target_class:", target_class_int)
    one_hot = F.one_hot(target_class_int, num_classes=logits.shape[1]).float()
    print("one_hot:", one_hot)

    # Backward from sum of target class scores (equivalent to batched backprop)
    logits_target = (logits * one_hot).sum()
    model.zero_grad()
    logits_target.backward()

    # calculate gradients, weights alpha
    gradients = feature_map.grad  # Shape: (B, C, H, W)
    weights_alpha = torch.mean(gradients, dim=[2, 3])  # Shape: (B, C)

    # weighted combination Σ_k α_k^c * A^k for each image
    num_channels = feature_map.shape[1]
    grad_cam_original = torch.zeros(
        (batch_size, feature_map.shape[2], feature_map.shape[3])
    )  # (B, H, W)
    for k in range(num_channels):
        grad_cam_original += (
            weights_alpha[:, k].unsqueeze(1).unsqueeze(2) * feature_map[:, k, :, :]
        )

    # ReLU grad_cam
    grad_cam_original = F.relu(grad_cam_original)

    # upsample to input size
    grad_cam_upsampled = F.interpolate(
        grad_cam_original.unsqueeze(1),
        size=(input_tensor.shape[2], input_tensor.shape[3]),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)
    print("grad_cam_upsampled:", grad_cam_upsampled.shape)

    # normalize grad cam
    gradcam_original_normalized = normalize_batch_gradcam(grad_cam_original, batch_size)
    gradcam_upsampled_normalized = normalize_batch_gradcam(
        grad_cam_upsampled, batch_size
    )

    # freeze features again
    model.freeze_features()

    return (
        y_pred,  # (B,) predicted classes
        y_prob,  # (B,) predicted classes
        gradcam_original_normalized.detach().numpy(),  # (B, H_feat, W_feat) original resolution
        gradcam_upsampled_normalized.detach().numpy(),  # (B, 28, 28) upsampled to input size
    )
