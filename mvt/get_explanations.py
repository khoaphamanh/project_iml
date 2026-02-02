from .custom_model_mvt import EfficientNetB0Autoencoder
import torch
import torch.nn.functional as F
from captum.attr import IntegratedGradients
import numpy as np


def normalize_batch_gradcam(tensor, batch_size):
    """Normalize each sample in batch to [0, 1] independently."""
    flat = tensor.view(batch_size, -1)
    min_vals = flat.min(dim=1, keepdim=True)[0].view(batch_size, 1, 1)
    max_vals = flat.max(dim=1, keepdim=True)[0].view(batch_size, 1, 1)
    return (tensor - min_vals) / (max_vals - min_vals + 1e-8)


def get_gradcam(
    model: EfficientNetB0Autoencoder,
    input_tensor: torch.Tensor,
    target_class_int=None,
):

    # model in eval mode
    model.eval()
    batch_size = input_tensor.shape[0]

    # forward pass
    logits, logits_mask, feature_map = model.get_feature_maps(input_tensor)
    logits_mask = torch.sigmoid(logits_mask)
    logits_mask = normalize_batch_gradcam(logits_mask, batch_size)
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
    y_prob = torch.softmax(logits.cpu(), dim=1).amax(dim=1).detach().numpy()

    # zero gradients
    model.zero_grad()

    # Create one-hot encoding for target classes: sum over batch for efficiency
    one_hot = F.one_hot(target_class_int, num_classes=logits.shape[1]).float()

    # Backward from sum of target class scores (equivalent to batched backprop)
    logits_target = (logits * one_hot).sum()
    model.zero_grad()
    logits_target.backward()

    # calculate gradients, weights alpha
    gradients = feature_map.grad  # Shape: (B, C, H, W)
    weights_alpha = torch.mean(gradients, dim=[2, 3])  # Shape: (B, C)

    # convert weights alpha and feature map to cpu
    weights_alpha = weights_alpha.cpu()
    feature_map = feature_map.cpu()

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

    # normalize grad cam
    gradcam_original_normalized = normalize_batch_gradcam(grad_cam_original, batch_size)
    gradcam_upsampled_normalized = normalize_batch_gradcam(
        grad_cam_upsampled, batch_size
    )

    return (
        y_pred,
        y_prob,
        gradcam_original_normalized.detach().numpy(),  # (B, H_feat, W_feat) original resolution
        gradcam_upsampled_normalized.detach().numpy(),  # (B, 28, 28) upsampled to input size
        logits_mask.detach().cpu().numpy(),
    )


def compute_saliency_map(
    model: EfficientNetB0Autoencoder, input_tensor, target_class=None
):
    model.eval()
    input_tensor.requires_grad_()
    # model.unfreeze_features()

    # Forward pass
    logits, _ = model(input_tensor)  # Shape: (B, num_classes)
    batch_size = input_tensor.shape[0]

    if target_class is None:
        target_class = logits.argmax(dim=1)  # Shape: (B,)

    # Create a mask to pick the target class score for each batch item
    # This is more efficient for batches than indexing in a loop
    one_hot = torch.zeros_like(logits)
    one_hot.scatter_(1, target_class.unsqueeze(1), 1.0)

    model.zero_grad()
    # Backward from the sum of target scores (gradients stay separated per batch item)
    (logits * one_hot).sum().backward()

    # Get absolute gradients: (B, C, H, W)
    grads = input_tensor.grad.data.abs()

    # Take max across color channels (dim=1) -> Result: (B, H, W)
    saliency, _ = torch.max(grads, dim=1)

    H, W = saliency.shape[1], saliency.shape[2]  #

    # --- NEW: PRINT INDEX OF MAX AND MIN PIXELS ---
    for i in range(batch_size):
        # Find flat indices
        max_idx = torch.argmax(saliency[i])
        min_idx = torch.argmin(saliency[i])

        # Convert to 2D coordinates (row, col)
        max_coord = (max_idx.item() // W, max_idx.item() % W)
        min_coord = (min_idx.item() // W, min_idx.item() % W)

        # print(f"Image {i} Saliency - Max at: {max_coord}, Min at: {min_coord}")

    # Normalize per image in the batch to [0, 1]
    saliency = normalize_batch_gradcam(saliency, batch_size)

    return saliency.detach().cpu().numpy()


def compute_integrated_gradients(model, input_tensor, target_class=None):
    model.eval()

    # Wrapper to return ONLY the classification logits
    def forward_wrapper(input):
        logits, _ = model(input)
        return logits

    ig = IntegratedGradients(forward_wrapper)
    batch_size = input_tensor.shape[0]

    if target_class is None:
        output, _ = model(input_tensor)
        target_class = output.argmax(dim=1)

    baseline = torch.zeros_like(input_tensor)

    attributions, delta = ig.attribute(
        input_tensor,
        baselines=baseline,
        target=target_class,
        return_convergence_delta=True,
        internal_batch_size=1,
        n_steps=25,
    )

    ig_map = torch.abs(attributions).sum(dim=1)

    H, W = ig_map.shape[1], ig_map.shape[2]  #
    for i in range(batch_size):
        max_idx = torch.argmax(ig_map[i])
        min_idx = torch.argmin(ig_map[i])

        max_coord = (max_idx.item() // W, max_idx.item() % W)
        min_coord = (min_idx.item() // W, min_idx.item() % W)

        # print(f"Image {i} Int. Gradients - Max at: {max_coord}, Min at: {min_coord}")

    ig_map = normalize_batch_gradcam(ig_map, batch_size)

    return ig_map.detach().cpu().numpy()
