"""
Configuração centralizada do projeto de Esteganografia GAN.

Este arquivo contém todos os hiperparâmetros, constantes e configurações
utilizadas em treino, teste e validação.
"""
import torch

# ====================
# Arquitetura do Modelo
# ====================
DATA_DEPTH = 1
"""Profundidade do canal de dados secretos."""

HIDDEN_SIZE = 32
"""Tamanho das camadas escondidas no Encoder/Decoder/Discriminador."""

IMAGE_SIZE = 360
"""Tamanho padrão das imagens (altura e largura em pixels)."""


# ====================
# Hiperparâmetros de Treinamento
# ====================
LEARNING_RATE = 0.001
"""Taxa de aprendizado para os otimizadores."""

BATCH_SIZE = 8
"""Tamanho do batch para treinamento."""

NUM_EPOCHS = 100
"""Número padrão de épocas de treinamento."""


# ====================
# Pesos das Funções de Loss
# ====================
BETA = 0.75
"""Peso da perda de qualidade de imagem (SSIM). 
Controla o quão similar a imagem esteganográfica deve ser à original."""

GAMMA = 0.1
"""Peso da perda adversarial (para enganar o Discriminador).
Controla o quão 'realista' a imagem codificada deve parecer."""

PADDING_WEIGHT = 0.01
"""Peso para bits de padding na loss de recuperação de dados.
Bits de padding têm menor importância que bits válidos da mensagem."""


# ====================
# Configuração de Mensagens
# ====================
MAX_MESSAGE_LENGTH = 100
"""Comprimento máximo de caracteres permitido em uma mensagem."""

BITS_PER_CHAR = 8
"""Número de bits usados para codificar cada caractere (padrão ASCII/UTF-8)."""

MAX_BITS = MAX_MESSAGE_LENGTH * BITS_PER_CHAR
"""Número máximo total de bits que podem ser codificados em uma imagem."""


# ====================
# Caminhos e Arquivos
# ====================
FRASES_PATH = "frases_geradas.txt"
"""Caminho para o arquivo de frases geradas para treinamento."""

DEFAULT_SAVE_PATH = "checkpoints"
"""Pasta padrão para salvar checkpoints e modelos."""


# ====================
# Device e Hardware
# ====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""Device automático: CUDA se disponível, senão CPU."""


# ====================
# Configurações de Validação
# ====================
VALIDATION_SPLIT = 0.1
"""Proporção de dados reservados para validação (10%)."""

SAVE_BEST_ONLY = True
"""Se True, salva apenas o modelo com melhor desempenho em validação."""


# ====================
# Logging e Debug
# ====================
VERBOSE = True
"""Ativa prints detalhados durante treinamento/teste."""

LOG_INTERVAL = 10
"""Intervalo de batches para logging durante treinamento."""


# ====================
# Funções Auxiliares
# ====================
def print_config():
    """Imprime todas as configurações atuais."""
    print("=" * 60)
    print("CONFIGURAÇÃO DO PROJETO")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Image Size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Hidden Size: {HIDDEN_SIZE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Max Message Length: {MAX_MESSAGE_LENGTH} chars ({MAX_BITS} bits)")
    print(f"Loss Weights - BETA: {BETA}, GAMMA: {GAMMA}")
    print("=" * 60)


if __name__ == "__main__":
    # Teste: imprimir configurações
    print_config()
