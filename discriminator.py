import torch
from torch import nn

class BasicDiscriminator(nn.Module):
    """
    Objetivo: Classificar se uma imagem é Real (Classe 1) ou Falsa (Classe 0).
    Design: PatchGAN
    
    Input: (N, 3, H, W)
    Output: (N, 2, H', W') <-- Logits de 2 classes para cada patch
    """

    def _conv_block(self, in_channels, out_channels, stride=2):
        """Bloco convolucional com Stride para Downsampling"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        self.model = nn.Sequential(
            # Input: (N, 3, 360, 360)
            
            nn.Conv2d(3, hidden_size, kernel_size=3, stride=2, padding=1), # -> 180x180
            nn.LeakyReLU(inplace=True),

            self._conv_block(hidden_size, hidden_size * 2, stride=2),      # -> 90x90
            self._conv_block(hidden_size * 2, hidden_size * 4, stride=2),  # -> 45x45
            self._conv_block(hidden_size * 4, hidden_size * 8, stride=2),  # -> 23x23
            
            nn.Conv2d(hidden_size * 8, 2, kernel_size=3, stride=1, padding=1),

        )

    def forward(self, image):
        # A saída agora é (N, 2, H', W'), pronta para CrossEntropyLoss
        return self.model(image)