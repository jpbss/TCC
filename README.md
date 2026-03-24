# STEGS2GAN - Um Modelo de Aprendizado Profundo para Esteganografia e Esteganálise em Imagens Digitais utilizando Redes Adversárias Generativas 🔐🖼️

Modelo que utiliza Redes Adversariais Generativas (GANs) para ocultar mensagens de texto em imagens, mantendo alta qualidade visual e imperceptibilidade.

## 📋 Índice

- [Descrição](#descrição)
- [Características](#características)
- [Instalação](#instalação)
- [Uso](#uso)
- [Arquitetura](#arquitetura)
- [Resultados](#resultados)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Licença](#licença)

## 📖 Descrição

Este projeto implementa um sistema completo de esteganografia usando técnicas de Deep Learning. O sistema é capaz de:

- **Ocultar** mensagens de texto (até 100 caracteres) dentro de imagens
- **Manter** alta qualidade visual (PSNR > 30dB, SSIM > 0.95)
- **Recuperar** as mensagens com alta precisão
- **Resistir** a análise visual humana

O sistema utiliza uma arquitetura GAN composta por:
- **Encoder (Gerador)**: Esconde a mensagem na imagem
- **Decoder**: Extrai a mensagem escondida
- **Discriminator**: Garante que imagens esteganográficas sejam indistinguíveis de imagens normais

## ✨ Características

- ✅ Arquitetura baseada em blocos residuais (ResNet-inspired)
- ✅ Dois tipos de discriminadores: BasicDiscriminator (PatchGAN) e SRNet
- ✅ Loss combinada: SSIM + Adversarial + Recuperação de dados
- ✅ Sistema de checkpoint para retomar treinamento
- ✅ Validação automática e salvamento do melhor modelo
- ✅ Suporte para Label Smoothing e Learning Rate Ratio
- ✅ Métricas de qualidade: PSNR e SSIM
- ✅ Interface de linha de comando (CLI)

## 🚀 Instalação

### Requisitos

- Python 3.8+
- CUDA 12.6+ (opcional, para GPU)

### Passos

1. Clone o repositório:
```bash
git clone https://github.com/jpbss/TCC.git
cd TCC
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

## 💻 Uso

### 1. Preparar Dados

Organize suas imagens em duas pastas:
```
dataset/
├── train/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
└── val/
    ├── img1.jpg
    ├── img2.jpg
    └── ...
```

### 2. Gerar Frases de Treinamento

```bash
python frases.py
```

Isso criará o arquivo `frases_geradas.txt` com frases aleatórias para treinamento.

### 3. Treinar o Modelo

**Com discriminador básico (PatchGAN):**
```bash
python train.py --train-dir ./dataset/train --val-dir ./dataset/val --disc basic --epochs 100
```

**Com discriminador SRNet:**
```bash
python train.py --train-dir ./dataset/train --val-dir ./dataset/val --disc srnet --epochs 100
```

**Com melhorias anti-mode-collapse:**
```bash
python train.py \
  --train-dir ./dataset/train \
  --val-dir ./dataset/val \
  --disc basic \
  --epochs 100 \
  --label-smoothing 0.1 \
  --lr-ratio 0.5
```

### 4. Testar com uma Imagem

**Teste único:**
```bash
python test.py -i caminho/para/imagem.jpg -m "Mensagem secreta" --model best --disc basic
```

**Teste em lote:**
```bash
python test_batch.py --test-dir ./dataset/test --model best --disc basic
```

### 5. Visualizar Curvas de Treinamento

Para plotar as curvas de treinamento a partir dos logs:

```bash
python curva_treinamento.py
```

O script irá gerar gráficos com:
- Loss de recuperação de dados
- Loss de qualidade de imagem (1-SSIM)
- Loss adversarial (Gerador vs Discriminador)
- Loss geral combinada

Os gráficos são salvos no diretório `results/` em alta resolução (300 DPI).

## 🏗️ Arquitetura

### Encoder (Gerador)
```
Input: Imagem (3, 360, 360) + Dados (1, 360, 360)
    ↓
[Feature Extraction] → Conv + BN + LeakyReLU
    ↓
[Concatenação] → Features + Dados Secretos
    ↓
[Blocos Residuais] × 3
    ↓
[Conv de Saída] → Sigmoid
    ↓
Output: Imagem Esteganográfica (3, 360, 360)
```

### Decoder
```
Input: Imagem Esteganográfica (3, 360, 360)
    ↓
[Conv de Entrada] → Conv + BN + LeakyReLU
    ↓
[Blocos Residuais] × 3
    ↓
[Conv de Saída] → Sigmoid
    ↓
Output: Dados Recuperados (1, 360, 360)
```

### Discriminator (PatchGAN)
```
Input: Imagem (3, 360, 360)
    ↓
[Conv Layers] × 4 (com stride=2 para downsampling)
    ↓
[Conv Final] → 2 classes (Real/Fake)
    ↓
Output: Logits (N, 2, H', W')
```

## 📊 Resultados

### Métricas de Qualidade Visual

| Métrica | Valor Médio | Descrição |
|---------|-------------|-----------|
| PSNR    | > 30 dB     | Relação sinal-ruído de pico |
| SSIM    | > 0.95      | Similaridade estrutural |

### Capacidade

- **Tamanho da mensagem**: Até 100 caracteres (800 bits)
- **Tamanho da imagem**: 360×360 pixels
- **Taxa de bits**: ~0.006 bits/pixel

### Acurácia de Recuperação

- Taxa de recuperação correta: > 95% em condições ideais

## 📁 Estrutura do Projeto

```
TCC/
├── config.py              # Configurações centralizadas
├── encoder.py             # Arquitetura do Encoder
├── decoder.py             # Arquitetura do Decoder
├── discriminator.py       # Discriminador PatchGAN
├── SRNet.py              # Discriminador SRNet
├── dataloader.py         # Carregamento de dados
├── utils.py              # Funções utilitárias
├── train.py              # Script de treinamento
├── test.py               # Teste em imagem única
├── test_batch.py         # Teste em lote
├── curva_treinamento.py  # Visualização de curvas de treinamento
├── frases.py             # Gerador de frases
├── requirements.txt      # Dependências
└── README.md            # Este arquivo
```

## 🔧 Configuração

Edite `config.py` para ajustar hiperparâmetros:

```python
# Arquitetura
DATA_DEPTH = 1
HIDDEN_SIZE = 32
IMAGE_SIZE = 360

# Treinamento
LEARNING_RATE = 0.001
BATCH_SIZE = 8
NUM_EPOCHS = 100

# Loss Weights
BETA = 0.75    # Peso da loss de qualidade (SSIM)
GAMMA = 0.1    # Peso da loss adversarial

# Mensagens
MAX_MESSAGE_LENGTH = 100
```

## 📝 Exemplos de Uso

### Exemplo 1: Treinamento Completo

```bash
# 1. Gerar frases
python frases.py

# 2. Treinar
python train.py --train-dir ./dataset/train --val-dir ./dataset/val --epochs 50

# 3. Testar
python test.py -i test_image.jpg -m "Olá Mundo!" --model best
```

### Exemplo 2: Retomar Treinamento

O treinamento é automaticamente retomado se houver um checkpoint:

```bash
python train.py --train-dir ./dataset/train --val-dir ./dataset/val --epochs 100
```

## 🤝 Contribuindo

Contribuições são bem-vindas! Por favor:

1. Faça um fork do projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📄 Licença

Este projeto foi desenvolvido como Trabalho de Conclusão de Curso (TCC).

## 📚 Referências

1. Zhu et al. (2018) - "HiDDeN: Hiding Data With Deep Networks"
2. Hayes & Danezis (2017) - "Generating Steganographic Images via Adversarial Training"
3. Baluja (2017) - "Hiding Images in Plain Sight: Deep Steganography"
4. Goodfellow et al. (2014) - "Generative Adversarial Networks"

## 👤 Autor

João Pedro - [jpbss](https://github.com/jpbss)

## 🙏 Agradecimentos

- Comunidade PyTorch
- Autores das bibliotecas utilizadas
- Orientador(a) do TCC

---

**Nota**: Este é um projeto acadêmico desenvolvido para fins educacionais. Para uso em produção, considere aspectos adicionais de segurança e robustez.