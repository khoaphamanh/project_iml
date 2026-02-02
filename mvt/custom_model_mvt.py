from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import torch
from torch import nn


class EfficientNetB0Autoencoder(torch.nn.Module):
    def __init__(self, num_classes=3, n_channels_encoder_base=256, n=3):
        super(EfficientNetB0Autoencoder, self).__init__()
        self.encoder = self.get_features_from_efficientnet_b0()

        # freeze features
        self.freeze_features()
        self.unfreeze_last_n_blocks(n=n)

        c1 = n_channels_encoder_base
        c2 = max(n_channels_encoder_base // 2, 32)
        c3 = max(n_channels_encoder_base // 4, 16)
        c4 = max(n_channels_encoder_base // 8, 8)
        c5 = max(n_channels_encoder_base // 16, 4)

        # Bottleneck: 7x7 -> 1x1
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))

        # Upsample back: 1x1 -> 7x7
        #  = nn.ConvTranspose2d(
        #     1280, 1280, kernel_size=7, stride=1, padding=0
        # )
        # self.upsample = nn.Upsample(size=(7, 7), mode="bilinear", align_corners=True)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1280, c1, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(),  # inplace=True
            nn.Conv2d(c1, c1, kernel_size=3, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(),
            nn.ConvTranspose2d(c1, c2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(),
            nn.Conv2d(c2, c2, kernel_size=3, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(),
            nn.ConvTranspose2d(c2, c3, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c3),
            nn.ReLU(),
            nn.Conv2d(c3, c3, kernel_size=3, padding=1),
            nn.BatchNorm2d(c3),
            nn.ReLU(),
            nn.ConvTranspose2d(c3, c4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c4),
            nn.ReLU(),
            nn.Conv2d(c4, c4, kernel_size=3, padding=1),
            nn.BatchNorm2d(c4),
            nn.ReLU(),
            nn.ConvTranspose2d(c4, c5, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c5),
            nn.ReLU(),
            nn.Conv2d(c5, c5, kernel_size=3, padding=1),
            nn.BatchNorm2d(c5),
            nn.ReLU(),
            nn.Conv2d(c5, out_channels=1, kernel_size=3, padding=1),
        )

        self.classifier = nn.Sequential(
            torch.nn.Dropout(p=0.2, inplace=True),
            torch.nn.Linear(in_features=1280, out_features=num_classes),
        )

    def unfreeze_last_n_blocks(self, n: int = 1):
        """
        Unfreeze only the last n "children" blocks of self.encoder,
        freeze all earlier ones.

        Note: EfficientNet's .features is a Sequential of stages/blocks. Unfreezing the
        last 1-2 blocks is a common fine-tuning approach.
        """
        blocks = list(self.encoder.children())
        if n <= 0:
            # freeze everything
            self.freeze_features()
            return
        if n > len(blocks):
            n = len(blocks)

        # Freeze earlier blocks
        for b in blocks[:-n]:
            for p in b.parameters():
                p.requires_grad = False

        # Unfreeze last n blocks
        for b in blocks[-n:]:
            for p in b.parameters():
                p.requires_grad = True

    def freeze_features(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def get_features_from_efficientnet_b0(self):
        weights = EfficientNet_B0_Weights.DEFAULT
        efficientnet_b0_model = efficientnet_b0(weights=weights)
        layer_features = efficientnet_b0_model.features
        return layer_features

    def forward(self, x):
        # get feature
        x_feature = self.encoder(x)

        # x_feature_decoded = self.upsample(x_latent)
        # recontruction
        mask_logits = self.decoder(x_feature)

        # classification
        x_latent = self.avgpool(x_feature)
        x_latent = torch.flatten(x_latent, 1)
        y_pred = self.classifier(x_latent)
        return y_pred, mask_logits

    def get_feature_maps(self, x):
        # get feature
        x_feature = self.encoder(x)

        # x_feature_decoded = self.upsample(x_latent)
        # recontruction
        mask_logits = self.decoder(x_feature)

        # classification
        x_latent = self.avgpool(x_feature)
        x_latent = torch.flatten(x_latent, 1)
        y_pred = self.classifier(x_latent)
        return y_pred, mask_logits, x_feature


if __name__ == "__main__":
    model = EfficientNetB0Autoencoder()

    from torchinfo import summary

    x = torch.randn(10, 3, 224, 224)
    summary(model, x)

    y_pred, mask_logits = model(x)
    print(y_pred.shape, mask_logits.shape)
