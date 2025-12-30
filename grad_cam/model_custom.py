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
