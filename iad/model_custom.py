from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import torch
from torch import nn


class EfficientNetB0Modified(torch.nn.Module):
    def __init__(self, num_classes=10):
        super(EfficientNetB0Modified, self).__init__()

        # feature layer from efficientnet_b0
        self.efficient_net_b0_features = self.get_features_from_efficientnet_b0()

        # freeze features
        self.freeze_features()

        # adaptive pooling
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))

        # classifier head
        self.dropout = torch.nn.Dropout(p=0.2, inplace=True)
        self.classifier = torch.nn.Linear(in_features=1280, out_features=num_classes)

    def get_features_from_efficientnet_b0(self):
        weights = EfficientNet_B0_Weights.DEFAULT
        efficientnet_b0_model = efficientnet_b0(weights=weights)
        layer_features = efficientnet_b0_model.features
        return layer_features

    def forward(self, x):
        x = self.efficient_net_b0_features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def get_feature_maps(self, x):
        self.eval()
        feature_map = self.efficient_net_b0_features(x)
        x = self.avgpool(feature_map)
        x = torch.flatten(x, 1)
        out = self.classifier(x)
        return feature_map, out

    def freeze_features(self):
        for param in self.efficient_net_b0_features.parameters():
            param.requires_grad = False

    def unfreeze_features(self):
        for param in self.efficient_net_b0_features.parameters():
            param.requires_grad = True


class EfficientNetB0Autoencoder(torch.nn.Module):
    def __init__(self, out_channels=3, base_channels=256):
        super(EfficientNetB0Autoencoder, self).__init__()
        self.encoder = self.get_features_from_efficientnet_b0()
        self.freeze_encoder()

        c1 = base_channels
        c2 = max(base_channels // 2, 32)
        c3 = max(base_channels // 4, 16)
        c4 = max(base_channels // 8, 8)
        c5 = max(base_channels // 16, 4)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1280, c1, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c1, c2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, c2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c2, c3, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3, c3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c3, c4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c4, c4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c4, c5, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c5, c5, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c5, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def get_features_from_efficientnet_b0(self):
        weights = EfficientNet_B0_Weights.DEFAULT
        efficientnet_b0_model = efficientnet_b0(weights=weights)
        return efficientnet_b0_model.features

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon

    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False


if __name__ == "__main__":

    import os
    from torchinfo import summary

    torch.hub.set_dir(os.path.join(os.getcwd(), "torch_cache111111111111"))

    # Example input: batch_size=2, channels=3, height=224, width=224
    input_tensor = torch.randn(2, 3, 224, 224)
    input_tensor = input_tensor.clamp(0, 1)

    clf = EfficientNetB0Modified(num_classes=10)
    logits = clf(input_tensor)
    print("Classifier output shape:", logits.shape)

    autoencoder = EfficientNetB0Autoencoder(out_channels=3, base_channels=128)
    summary(autoencoder, input_size=(10, 3, 224, 224))
    recon = autoencoder(input_tensor)

    print("Autoencoder output shape:", recon.shape)
