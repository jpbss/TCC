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


class BasicEncoder(nn.Module):
    """
    Encoder (Gerador).
    
    Input: (N, 3, H, W) [Imagem], (N, D, H, W) [Dados]
    Output: (N, 3, H, W) [Stego Image]
    """

    add_image = False # Flag para modo residual

    def _conv2d(self, canais_entrada, canais_saida):
        return nn.Conv2d(
            in_channels=canais_entrada,
            out_channels=canais_saida,
            kernel_size=3,
            padding=1
        )

    def construir_modelo(self):
        # 1. Rede de extração de features da Imagem
        self.features = nn.Sequential(
            self._conv2d(3, self.tamanho_dados_escondidos),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.tamanho_dados_escondidos),
        )
        
        # 2. Camada de entrada para os dados + features
        self.conv_in = nn.Sequential(
            self._conv2d(self.tamanho_dados_escondidos + self.profundidade_dados, self.tamanho_dados_escondidos),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.tamanho_dados_escondidos),
        )
        
        # 3. Módulos Residuais para processamento profundo
        self.residual_blocks = nn.Sequential(
            ResidualBlock(self.tamanho_dados_escondidos),
            ResidualBlock(self.tamanho_dados_escondidos),
            ResidualBlock(self.tamanho_dados_escondidos)
        )
        
        # 4. Camada de saída
        self.conv_out = nn.Sequential(
            self._conv2d(self.tamanho_dados_escondidos, 3),
            nn.Sigmoid(), # Sigmoid para manter os valores entre 0-1
        )
        
        # Registra os módulos
        return [self.features, self.conv_in, self.residual_blocks, self.conv_out]

    def __init__(self, profundidade_dados, tamanho_dados_escondidos):
        super().__init__()
        self.version = '2_residual_gan'
        self.profundidade_dados = profundidade_dados
        self.tamanho_dados_escondidos = tamanho_dados_escondidos
        
        # Constrói e registra os módulos
        self.features, self.conv_in, self.residual_blocks, self.conv_out = self.construir_modelo()

    def forward(self, image, data):
        # 1. Extrai features da imagem
        features = self.features(image)
        
        # 2. Concatena features com os dados secretos
        combined = torch.cat([features, data], dim=1)

        # 3. Passa pela rede
        x = self.conv_in(combined)
        x = self.residual_blocks(x)
        output = self.conv_out(x)

        if self.add_image:
            return image + output
        else:
            return output