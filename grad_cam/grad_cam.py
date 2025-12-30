from .model_custom import EfficientNetB0Modified
import torch
import torch.nn.functional as F


def get_gradcam(
    model: EfficientNetB0Modified,
    input_tensor: torch.Tensor,
    target_class_int=None,
):

    # model in eval mode
    model.eval()
    batch_size = input_tensor.shape[0]
    layer_feature_efficientnet_b0 = model.efficient_net_b0_features

    # unfreeze features
    model.unfreeze_features()

    # call back hook to get gradients
    gradients_captured = {}

    def callback_hook(module, grad_input, grad_output):
        # grad_output
        gradients_captured["feature_map"] = grad_output[0].detach()

    # register hook
    hook = layer_feature_efficientnet_b0.register_full_backward_hook(callback_hook)

    # forward pass
    feature_map, logits = model.get_feature_maps(input_tensor)

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
    one_hot = torch.zeros_like(logits)  # (B, num_classes)
    for i in range(batch_size):
        one_hot[i, target_class_int[i]] = 1.0

    # Backward from sum of target class scores (equivalent to batched backprop)
    logits_target = (logits * one_hot).sum()
    logits_target.backward(retain_graph=True)

    # remove the hook
    hook.remove()

    # calculate gradients, weights alpha
    gradients = gradients_captured["feature_map"]  # Shape: (B, C, H, W)
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

    # normalize grad cam
    gradcam_original_normalized = []
    gradcam_upsampled_normalized = []

    for i in range(batch_size):
        # Normalize original resolution
        cam_orig = grad_cam_original[i]
        cam_orig = cam_orig - cam_orig.min()
        cam_orig = cam_orig / (cam_orig.max() + 1e-8)
        gradcam_original_normalized.append(cam_orig)

        # Normalize upsampled
        cam_up = grad_cam_upsampled[i]
        cam_up = cam_up - cam_up.min()
        cam_up = cam_up / (cam_up.max() + 1e-8)
        gradcam_upsampled_normalized.append(cam_up)

    # Stack into tensors
    gradcam_original_normalized = torch.stack(
        gradcam_original_normalized
    )  # (B, H_feat, W_feat)
    gradcam_upsampled_normalized = torch.stack(
        gradcam_upsampled_normalized
    )  # (B, 224, 224)

    # freeze features again
    model.freeze_features()

    return (
        y_pred,  # (B,) predicted classes
        y_prob,  # (B,) predicted classes
        gradcam_original_normalized.detach().numpy(),  # (B, H_feat, W_feat) original resolution
        gradcam_upsampled_normalized.detach().numpy(),  # (B, 28, 28) upsampled to input size
    )
