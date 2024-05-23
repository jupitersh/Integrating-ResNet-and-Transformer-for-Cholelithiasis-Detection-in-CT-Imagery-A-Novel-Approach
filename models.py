import math
import torch
from torch import nn
from torchvision.models.resnet import resnet18, ResNet18_Weights

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class ResTransformer(nn.Module):
    def __init__(self, dropout_prob_res=0, dropout_prob_pe=0.1, dropout_prob_fc=[0, 0]):
        super(ResTransformer, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(1, 1))

        self.resnet18 = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        self.resnet18.fc = nn.Sequential(
            nn.Dropout(p=dropout_prob_res),
            nn.Linear(self.resnet18.fc.in_features, 128)
        )

        self.positional_encoding = PositionalEncoding(d_model=128, dropout=dropout_prob_pe)
        transformer_layer = nn.TransformerEncoderLayer(d_model=128, nhead=8, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=2)

        self.fc = nn.Sequential(
            nn.Dropout(dropout_prob_fc[0]),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_prob_fc[1]),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        batch_size, slices, C, H, W = x.size(0), x.size(1), x.size(2), x.size(3), x.size(4)

        resnet_output = torch.zeros(batch_size, slices, 128).to(x.device)

        for i in range(slices):
            layer = x[:, i, :, :, :]
            if torch.all(layer == 0):
                resnet_output[:, i, :] = torch.zeros(128).to(x.device)
            else:
                layer = self.conv1(layer)
                layer_feature = self.resnet18(layer)
                resnet_output[:, i, :] = layer_feature
                
        resnet_output = self.positional_encoding(resnet_output)

        transformer_output = self.transformer_encoder(resnet_output)

        self.output_for_tsne = transformer_output[:, -1, :]

        out = self.fc(self.output_for_tsne)

        return out

class ResCNN(nn.Module):
    def __init__(self, dropout_prob_res=0, dropout_prob_fc=[0, 0]):
        super(ResCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(1, 1))

        self.resnet18 = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        self.resnet18.fc = nn.Sequential(
            nn.Dropout(p=dropout_prob_res),
            nn.Linear(self.resnet18.fc.in_features, 128)
        )

        self.cnn_layers = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        self.adaptive_pool = nn.AdaptiveAvgPool1d(4)

        self.fc = nn.Sequential(
            nn.Dropout(dropout_prob_fc[0]),
            nn.Linear(32 *4, 64),
            nn.ReLU(),
            nn.Dropout(dropout_prob_fc[1]),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        batch_size, slices, C, H, W = x.size(0), x.size(1), x.size(2), x.size(3), x.size(4)

        resnet_output = torch.zeros(batch_size, slices, 128).to(x.device)

        for i in range(slices):
            layer = x[:, i, :, :, :]
            if torch.all(layer == 0):
                resnet_output[:, i, :] = torch.zeros(128).to(x.device)
            else:
                layer = self.conv1(layer)
                layer_feature = self.resnet18(layer)
                resnet_output[:, i, :] = layer_feature

        resnet_output = resnet_output.transpose(1, 2)

        cnn_output = self.cnn_layers(resnet_output)

        pooled_output = self.adaptive_pool(cnn_output)
        self.output_for_tsne = pooled_output.view(batch_size, -1)

        out = self.fc(self.output_for_tsne)

        return out
