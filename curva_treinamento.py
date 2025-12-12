import matplotlib.pyplot as plt
import numpy as np
import re

def parse_log_file(log_path):
    """Extrai épocas e losses do log de treinamento"""
    epochs = []
    train_losses = {'dados': [], 'img': [], 'adv_g': [], 'disc': []}
    val_losses = {'dados': [], 'img': [], 'adv_g': [], 'disc': [], 'geral': []}
    
    with open(log_path, 'r') as f:
        content = f.read()
    
    # Regex para validação (mais confiável)
    pattern = r'Epoch \[(\d+)/\d+\] Resumo da Validação:\s+\[Loss Dados\]: ([\d.]+)\s+\| \[Loss Imagem.*?\]: ([\d.]+)\s+\| \[Loss Adv\. G\]: ([\d.]+)\s+\| \[Loss Disc\]: ([\d.]+)\s+-> Loss GERAL.*?: ([\d.]+)'
    
    matches = re.findall(pattern, content)
    
    for match in matches:
        epoch, dados, img, adv, disc, geral = match
        epochs.append(int(epoch))
        val_losses['dados'].append(float(dados))
        val_losses['img'].append(float(img))
        val_losses['adv_g'].append(float(adv))
        val_losses['disc'].append(float(disc))
        val_losses['geral'].append(float(geral))
    
    return epochs, val_losses

def plot_training_curves(log_path, title, output_path):
    """Plota curvas de treinamento"""
    epochs, val = parse_log_file(log_path)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Loss Dados
    axes[0, 0].plot(epochs, val['dados'], 'b-', linewidth=2)
    axes[0, 0].set_title('Loss de Dados (Validação)', fontweight='bold')
    axes[0, 0].set_xlabel('Época')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(alpha=0.3)
    
    # Loss Imagem
    axes[0, 1].plot(epochs, val['img'], 'g-', linewidth=2)
    axes[0, 1].set_title('Loss de Imagem (1-SSIM)', fontweight='bold')
    axes[0, 1].set_xlabel('Época')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(alpha=0.3)
    
    # Loss Adversarial
    axes[1, 0].plot(epochs, val['adv_g'], 'r-', label='Gerador', linewidth=2)
    axes[1, 0].plot(epochs, val['disc'], 'orange', label='Discriminador', linewidth=2)
    axes[1, 0].axhline(y=0.693, color='gray', linestyle='--', label='Equilíbrio (ln2)')
    axes[1, 0].set_title('Loss Adversarial', fontweight='bold')
    axes[1, 0].set_xlabel('Época')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # Loss Geral
    axes[1, 1].plot(epochs, val['geral'], 'purple', linewidth=2)
    best_epoch = epochs[np.argmin(val['geral'])]
    best_loss = min(val['geral'])
    axes[1, 1].axvline(x=best_epoch, color='red', linestyle='--', 
                       label=f'Melhor: Época {best_epoch} (Loss={best_loss:.4f})')
    axes[1, 1].set_title('Loss Geral (Encoder+Decoder)', fontweight='bold')
    axes[1, 1].set_xlabel('Época')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Gráfico salvo: {output_path}")

# Uso
plot_training_curves('div_basic.txt', 'DIV2K + BasicDiscriminator', 
                     'results/training_curves_div2k_basic.png')
plot_training_curves('div_srnet.txt', 'DIV2K + SRNet', 
                     'results/training_curves_div2k_srnet.png')
plot_training_curves('coco_basic.txt', 'COCO + BasicDiscriminator', 
                     'results/training_curves_coco_basic.png')
plot_training_curves('coco_srnet.txt', 'COCO + SRNet', 
                     'results/training_curves_coco_srnet.png')