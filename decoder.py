import torch
from torch import nn

class ResidualBlock(nn.Module):
    """Bloco Residual: x -> Conv -> BN -> ReLU -> Conv -> BN -> (+) -> x """
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )
    
    def forward(self, x):
        # A mágica do bloco residual: soma a entrada com a saída
        return x + self.conv(x) 

class BasicDecoder(nn.Module):
    """
    Decoder melhorado com blocos residuais.

    Input: (N, 3, H, W)  <-- A imagem esteganográfica
    Output: (N, D, H, W) <-- Os dados secretos recuperados
    """

    def _conv2d(self, canais_entrada, canais_saida):
        return nn.Conv2d(
            in_channels=canais_entrada,
            out_channels=canais_saida,
            kernel_size=3,
            padding=1
        )

    def construir_modelo(self):
        # 1. Camada de entrada
        self.conv_in = nn.Sequential(
            self._conv2d(3, self.tamanho_dados_escondidos),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.tamanho_dados_escondidos),
        )
        
        # 2. Módulos Residuais para processamento profundo
        self.residual_blocks = nn.Sequential(
            ResidualBlock(self.tamanho_dados_escondidos),
            ResidualBlock(self.tamanho_dados_escondidos),
            ResidualBlock(self.tamanho_dados_escondidos)
        )
        
        # 3. Camada de saída
        self.conv_out = nn.Sequential(
            self._conv2d(self.tamanho_dados_escondidos, self.profundidade_dados),
            nn.Sigmoid() # Sigmoid para dados binários (saída 0-1)
        )

        return [self.conv_in, self.residual_blocks, self.conv_out]

    def __init__(self, profundidade_dados, tamanho_dados_escondidos):
        super().__init__()
        self.version = '2_residual_gan'
        self.profundidade_dados = profundidade_dados
        self.tamanho_dados_escondidos = tamanho_dados_escondidos

        self.conv_in, self.residual_blocks, self.conv_out = self.construir_modelo()

    def forward(self, x):
        x = self.conv_in(x)
        x = self.residual_blocks(x)
        x = self.conv_out(x)
        return x