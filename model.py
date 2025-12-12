""" This module creates SRNet model."""
import torch
from torch import Tensor
from torch import nn
from utils_srnet import Type1, Type2, Type3


class Srnet(nn.Module):
    """SRNet adaptado para treinamento adversarial com imagens RGB 360x360.
    
    Mudanças em relação ao SRNet original:
    - Input: 3 canais (RGB) ao invés de 1 (grayscale)
    - Input size: 360x360 ao invés de 256x256
    - Output: Espacial (N, 2, H, W) ao invés de global (N, 2)
    - Sem LogSoftmax (CrossEntropyLoss aplica internamente)
    """

    def __init__(self) -> None:
        """Constructor."""
        super().__init__()
        # Camada inicial adaptada para RGB (3 canais)
        self.type1s = nn.Sequential(
            Type1(3, 64),   # 3→64 (RGB input)
            Type1(64, 16)   # 64→16
        )
        
        # Blocos residuais (mantidos)
        self.type2s = nn.Sequential(
            Type2(16, 16),
            Type2(16, 16),
            Type2(16, 16),
            Type2(16, 16),
            Type2(16, 16),
        )
        
        # Downsampling (mantido)
        self.type3s = nn.Sequential(
            Type3(16, 16),   # 360→180
            Type3(16, 64),   # 180→90
            Type3(64, 128),  # 90→45
            Type3(128, 256), # 45→23 (aprox)
        )
        
        # Classificador espacial (ao invés de GAP + Dense)
        # Output: (N, 2, 23, 23) - compatível com PatchGAN
        self.classifier = nn.Conv2d(256, 2, kernel_size=3, stride=1, padding=1)

    def forward(self, inp: Tensor) -> Tensor:
        """Returns spatial logits for input RGB images.
        
        Args:
            inp (Tensor): input image tensor of shape (Batch, 3, 360, 360)
            
        Returns:
            Tensor: Logits of shape (Batch, 2, ~23, ~23)
                    Compatible with CrossEntropyLoss expecting (N, C, H, W)
        """
        out = self.type1s(inp)     # (N, 16, 360, 360)
        out = self.type2s(out)     # (N, 16, 360, 360)
        out = self.type3s(out)     # (N, 256, ~23, ~23)
        out = self.classifier(out) # (N, 2, ~23, ~23)
        
        # NÃO aplica softmax - CrossEntropyLoss faz isso
        return out


if __name__ == "__main__":
    # Teste com imagem RGB 360x360
    image = torch.randn((4, 3, 360, 360))  # Batch de 4 imagens RGB
    net = Srnet()
    output = net(image)
    print(f"Input shape:  {image.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected: (4, 2, H, W) onde H,W ≈ 23")
    
    # Teste de compatibilidade com CrossEntropyLoss
    target = torch.randint(0, 2, (4, output.shape[2], output.shape[3]))
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(output, target)
    print(f"\nTeste com CrossEntropyLoss: Loss = {loss.item():.4f}")

