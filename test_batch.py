import torch
import torch.nn as nn
import os
import numpy as np
import csv
import sys
from PIL import Image
from torchvision import transforms
import argparse 
from tqdm import tqdm
import matplotlib.pyplot as plt 
from pytorch_msssim import ssim 
from torchvision.transforms.functional import to_pil_image
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import json

from config import DATA_DEPTH, HIDDEN_SIZE, IMAGE_SIZE, BATCH_SIZE, DEVICE
from dataloader import DataLoader
from encoder import BasicEncoder
from decoder import BasicDecoder
from discriminator import BasicDiscriminator
from model import Srnet
from utils import text_to_tensor, tensor_to_text, MAX_BITS 
from piq import psnr

def calculate_psnr(img1, img2, data_range=1.0):
    """Calcula o PSNR entre dois tensores de imagem (formato NCHW)."""
    return psnr(img1, img2, data_range=data_range, reduction='mean').item()

def calculate_bit_accuracy(secret_tensor_flat, decoded_data_tensor, message_lengths):
    """Calcula a acurácia de bits percentual, ignorando o padding."""
    
    decoded_bits = decoded_data_tensor.view(secret_tensor_flat.size(0), -1)[:, :MAX_BITS]
    decoded_bits_rounded = (decoded_bits > 0.5).float()
    
    correct_bits = (decoded_bits_rounded == secret_tensor_flat).float()
    
    total_acc = 0
    for i in range(secret_tensor_flat.size(0)):
        length = message_lengths[i]
        if length > 0:
            total_acc += correct_bits[i, :length].mean().item()
        else:
            total_acc += 1.0 
            
    return (total_acc / secret_tensor_flat.size(0)) * 100

def calculate_message_accuracy(original_phrases_batch, decoded_data_tensor):
    """Calcula a acurácia de mensagem 'Sim/Não' (tudo ou nada)."""
    
    reconstructed_phrases = [tensor_to_text(t) for t in decoded_data_tensor]
    
    batch_message_success_count = 0
    successful_bool_list = []
    
    for original, reconstructed in zip(original_phrases_batch, reconstructed_phrases):
        is_success = original.strip() == reconstructed.strip()
        successful_bool_list.append(is_success)
        if is_success:
            batch_message_success_count += 1
            
    percent = (batch_message_success_count / len(original_phrases_batch)) * 100
    return percent, successful_bool_list, reconstructed_phrases

def save_comparison_images(cover_img, stego_img, output_path, img_id):
    """Salva comparação visual: Original | Stego | Diferença"""
    
    # Converter tensores para numpy (C, H, W) -> (H, W, C)
    cover_np = cover_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
    stego_np = stego_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
    
    # Diferença amplificada 10x
    diff_np = np.abs(cover_np - stego_np) * 10
    diff_np = np.clip(diff_np, 0, 1)
    
    # Criar figura com 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(cover_np)
    axes[0].set_title('Original', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(stego_np)
    axes[1].set_title('Stego', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(diff_np)
    axes[2].set_title('Diferença (10×)', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(cm, output_path, disc_type, model_suffix):
    """Plota e salva matriz de confusão"""
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=['Real', 'Stego'],
               yticklabels=['Real', 'Stego'],
               cbar_kws={'label': 'Contagem'},
               annot_kws={'size': 14, 'weight': 'bold'})
    
    plt.ylabel('Verdadeiro', fontsize=12, fontweight='bold')
    plt.xlabel('Predito', fontsize=12, fontweight='bold')
    plt.title(f'Matriz de Confusão - {disc_type.upper()} ({model_suffix})', 
              fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_roc_curve(fpr, tpr, roc_auc, output_path, disc_type, model_suffix):
    """Plota e salva curva ROC"""
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Aleatório (AUC = 0.500)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos (FPR)', fontsize=12, fontweight='bold')
    plt.ylabel('Taxa de Verdadeiros Positivos (TPR)', fontsize=12, fontweight='bold')
    plt.title(f'Curva ROC - {disc_type.upper()} ({model_suffix})', 
              fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def save_histogram(data, title, xlabel, filename, bins=20, range_min=None, range_max=None):
    """Salva histograma de distribuição"""
    
    plt.figure(figsize=(10, 6))
    data_range = (range_min, range_max) if range_min is not None else None
    plt.hist(data, bins=bins, color='steelblue', edgecolor='black', 
             range=data_range, alpha=0.7)
    
    mean_val = np.mean(data)
    median_val = np.median(data)
    std_val = np.std(data)
    
    plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, 
                label=f'Média: {mean_val:.4f}')
    plt.axvline(median_val, color='green', linestyle='dashed', linewidth=2, 
                label=f'Mediana: {median_val:.4f}')
    
    plt.title(f'{title}\n(μ={mean_val:.4f}, σ={std_val:.4f})', 
              fontsize=14, fontweight='bold')
    plt.xlabel(xlabel, fontsize=12, fontweight='bold')
    plt.ylabel('Contagem (Imagens)', fontsize=12, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Executa teste completo com todas as métricas e visualizações.')
    
    parser.add_argument('--image_dir', type=str, required=True, 
                        help='Diretório com as imagens de teste.')
    
    parser.add_argument('--phrases_file', type=str, default='frases_geradas.txt', 
                        help='Ficheiro de texto contendo as frases.')
    
    parser.add_argument('--results_dir', type=str, required=True, 
                        help='Pasta onde os resultados serão guardados.')
    
    parser.add_argument('--model_suffix', type=str, required=True, choices=['best', 'gan'], 
                        help="Sufixo do modelo: 'best' ou 'gan'.")
                        
    parser.add_argument('--model_path', type=str, required=True, 
                        help='Pasta onde os modelos (.pth) estão.')

    parser.add_argument('--disc', type=str, required=True, choices=['basic', 'srnet'],
                        help='Tipo de discriminador: "basic" ou "srnet".')
    
    parser.add_argument('--batch_size', type=int, default=8, 
                        help='Tamanho do batch.')
    
    parser.add_argument('--num_tests', type=int, default=0, 
                        help='Número de imagens a testar. Use 0 para todas.')
    
    parser.add_argument('--save_comparisons', type=int, default=10,
                        help='Número de comparações visuais a salvar.')
    
    args = parser.parse_args()
    
    device = DEVICE
    print(f"Usando o dispositivo: {device}")

    # --- Criar Pastas de Resultados ---
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Subpastas organizadas
    comparisons_dir = os.path.join(args.results_dir, 'comparisons')
    graphs_dir = os.path.join(args.results_dir, 'graphs')
    os.makedirs(comparisons_dir, exist_ok=True)
    os.makedirs(graphs_dir, exist_ok=True)
    
    summary_file = os.path.join(args.results_dir, f"summary_{args.disc}_{args.model_suffix}.txt")
    csv_file = os.path.join(args.results_dir, f"detailed_results_{args.disc}_{args.model_suffix}.csv")
    metrics_json = os.path.join(args.results_dir, f"metrics_{args.disc}_{args.model_suffix}.json")

    # --- Carregar Modelos ---
    print("Carregando modelos...")
    encoder = BasicEncoder(profundidade_dados=DATA_DEPTH, tamanho_dados_escondidos=HIDDEN_SIZE).to(device)
    decoder = BasicDecoder(profundidade_dados=DATA_DEPTH, tamanho_dados_escondidos=HIDDEN_SIZE).to(device)
    
    if args.disc == 'srnet':
        discriminator = Srnet().to(device)
        print("Usando discriminador: SRNet")
    else:
        discriminator = BasicDiscriminator(hidden_size=HIDDEN_SIZE).to(device)
        print("Usando discriminador: BasicDiscriminator (PatchGAN)")

    DISC_TYPE = args.disc
    LOSS_SUFFIX = 'ce_ssim'

    encoder_path = os.path.join(args.model_path, f'encoder_{args.model_suffix}_{DISC_TYPE}_{LOSS_SUFFIX}.pth')
    decoder_path = os.path.join(args.model_path, f'decoder_{args.model_suffix}_{DISC_TYPE}_{LOSS_SUFFIX}.pth')
    discriminator_path = os.path.join(args.model_path, f'discriminator_{args.model_suffix}_{DISC_TYPE}_{LOSS_SUFFIX}.pth')

    try:
        print(f"Carregando Encoder: {encoder_path}")
        encoder.load_state_dict(torch.load(encoder_path, map_location=device))
        
        print(f"Carregando Decoder: {decoder_path}")
        decoder.load_state_dict(torch.load(decoder_path, map_location=device))
        
        print(f"Carregando Discriminator: {discriminator_path}")
        discriminator.load_state_dict(torch.load(discriminator_path, map_location=device))
        
    except FileNotFoundError as e:
        print(f"Erro: Ficheiro de modelo não encontrado.")
        print(f"Ficheiro em falta: {e.filename}")
        exit()
    except Exception as e:
        print(f"Erro ao carregar modelos: {e}")
        exit()

    encoder.eval()
    decoder.eval()
    discriminator.eval()
    print("Modelos carregados e em modo de avaliação.\n")

    # --- Carregar Frases ---
    print(f"Carregando frases de '{args.phrases_file}'...")
    try:
        with open(args.phrases_file, 'r', encoding='utf-8') as f:
            all_phrases = [line.strip() for line in f.readlines() if line.strip()]
        print(f"{len(all_phrases)} frases carregadas.\n")
    except FileNotFoundError:
        print(f"ERRO: Ficheiro '{args.phrases_file}' não encontrado.")
        exit()

    # --- Preparar Dataloader ---
    print(f"Carregando imagens de '{args.image_dir}'...")
    
    limit = args.num_tests if args.num_tests > 0 else np.inf
    print(f"Testando {'todas as imagens' if limit == np.inf else f'{limit} imagens'}.\n")
        
    try:
        test_dataloader = DataLoader(args.image_dir, batch_size=args.batch_size, 
                                     shuffle=False, image_size=IMAGE_SIZE, 
                                     num_workers=2, limit=limit)
    except Exception as e:
        print(f"ERRO ao carregar imagens: {e}")
        exit()
        
    # --- Inicializar Listas de Métricas ---
    psnr_scores = []
    ssim_scores = []
    bit_acc_scores = []
    message_acc_scores = []
    
    # Para discriminador
    all_labels = []  # 0 = Real, 1 = Stego
    all_preds = []   # 0 ou 1
    all_probs = []   # Probabilidade de ser Stego
    
    global_image_count = 0
    saved_comparisons = 0

    # --- Cabeçalhos CSV ---
    csv_headers = ['Image_ID', 'Original_File', 'Stego_File', 
                   'Original_Phrase', 'Reconstructed_Phrase', 'Message_Match', 
                   'Bit_Accuracy_%', 'PSNR_dB', 'SSIM', 
                   'Disc_Real_Pred', 'Disc_Stego_Pred']
                   
    with open(csv_file, 'w', newline='', encoding='utf-8') as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(csv_headers)

    # --- Loop Principal de Teste ---
    print(f"{'='*60}")
    print(f"Iniciando teste em {len(test_dataloader)} batches")
    print(f"{'='*60}\n")
    
    pbar = tqdm(enumerate(test_dataloader), total=len(test_dataloader), 
                desc="Testando", unit="batch")
    
    with torch.no_grad():
        for batch_idx, (cover_image_batch, _) in pbar:
            cover_image = cover_image_batch.to(device)
            current_batch_size = cover_image.size(0)

            # --- 1. Preparar Mensagens ---
            batch_phrases_indices = np.random.choice(len(all_phrases), current_batch_size, replace=True)
            batch_phrases = [all_phrases[idx] for idx in batch_phrases_indices]
            
            secret_tensors_list = [text_to_tensor(p, device) for p in batch_phrases]
            secret_tensor_flat = torch.stack([r[0] for r in secret_tensors_list])
            message_lengths = [r[1] for r in secret_tensors_list]
            
            secret_data_tensor = torch.zeros(current_batch_size, DATA_DEPTH, 
                                            IMAGE_SIZE, IMAGE_SIZE, device=device)
            secret_data_tensor.view(current_batch_size, -1)[:, :MAX_BITS] = secret_tensor_flat

            # --- 2. Codificação ---
            stego_image = encoder(cover_image, secret_data_tensor)

            # --- 3. Decodificação ---
            decoded_data = decoder(stego_image)
            
            # --- 4. Discriminador ---
            pred_real = discriminator(cover_image)  # (N, 2, H, W) ou (N, 2)
            pred_stego = discriminator(stego_image)
            
            # Converter para probabilidades
            # Média espacial primeiro (se houver dimensões espaciais)
            if pred_real.dim() == 4:  # PatchGAN: (N, 2, H, W)
                pred_real = pred_real.mean(dim=[2, 3])  # (N, 2)
                pred_stego = pred_stego.mean(dim=[2, 3])
            
            # Softmax para probabilidades
            probs_real = torch.softmax(pred_real, dim=1)
            probs_stego = torch.softmax(pred_stego, dim=1)
            
            # Classe predita (0 ou 1)
            preds_real = torch.argmax(probs_real, dim=1).cpu().numpy()
            preds_stego = torch.argmax(probs_stego, dim=1).cpu().numpy()
            
            # Probabilidade da classe "Stego" (classe 1)
            prob_real_is_stego = probs_real[:, 1].cpu().numpy()
            prob_stego_is_stego = probs_stego[:, 1].cpu().numpy()
            
            # Guardar para métricas do discriminador
            # Labels: 0 = Real, 1 = Stego
            all_labels.extend([0] * current_batch_size)  # Imagens reais
            all_labels.extend([1] * current_batch_size)  # Imagens stego
            
            all_preds.extend(preds_real)
            all_preds.extend(preds_stego)
            
            all_probs.extend(prob_real_is_stego)
            all_probs.extend(prob_stego_is_stego)

            # --- 5. Message Accuracy ---
            _, successful_bool_list, reconstructed_phrases = calculate_message_accuracy(
                batch_phrases, decoded_data)
            
            # --- Loop por Imagem (Métricas e Salvamento) ---
            cover_imgs = torch.chunk(cover_image, current_batch_size)
            stego_imgs = torch.chunk(stego_image, current_batch_size)
            decoded_datas = torch.chunk(decoded_data, current_batch_size)
            secret_tensors = torch.chunk(secret_tensor_flat, current_batch_size)

            for j in range(current_batch_size):
                img_id = f"{global_image_count:05d}"
                
                original_phrase = batch_phrases[j]
                reconstructed_phrase = reconstructed_phrases[j]
                is_match = successful_bool_list[j]
                
                message_acc_scores.append(100.0 if is_match else 0.0)

                # Bit Accuracy
                current_bit_acc = calculate_bit_accuracy(
                    secret_tensors[j], decoded_datas[j], [message_lengths[j]])
                bit_acc_scores.append(current_bit_acc)
                
                # PSNR e SSIM
                img_cover = cover_imgs[j]
                img_stego = stego_imgs[j]
                current_psnr = calculate_psnr(img_cover, img_stego, data_range=1.0)
                current_ssim = ssim(img_cover, img_stego, data_range=1.0, 
                                   size_average=True).item()
                
                psnr_scores.append(current_psnr)
                ssim_scores.append(current_ssim)

                # Salvar imagens individuais
                original_file = f"{img_id}_original.png"
                stego_file = f"{img_id}_stego.png"
                
                to_pil_image(img_cover.squeeze(0).cpu()).save(
                    os.path.join(args.results_dir, original_file))
                to_pil_image(img_stego.squeeze(0).cpu()).save(
                    os.path.join(args.results_dir, stego_file))
                
                # Salvar comparações visuais (primeiras N imagens)
                if saved_comparisons < args.save_comparisons:
                    comparison_file = os.path.join(comparisons_dir, f'comparison_{img_id}.png')
                    save_comparison_images(img_cover, img_stego, comparison_file, img_id)
                    saved_comparisons += 1
                
                # Predições do discriminador para esta imagem
                pred_real_str = 'Real' if preds_real[j] == 1 else 'Stego'
                pred_stego_str = 'Stego' if preds_stego[j] == 1 else 'Real'
                
                # Escrever no CSV
                with open(csv_file, 'a', newline='', encoding='utf-8') as f_csv:
                    writer = csv.writer(f_csv)
                    writer.writerow([
                        img_id, original_file, stego_file,
                        original_phrase, reconstructed_phrase, is_match,
                        f"{current_bit_acc:.2f}",
                        f"{current_psnr:.4f}",
                        f"{current_ssim:.6f}",
                        pred_real_str,
                        pred_stego_str
                    ])
                
                global_image_count += 1
            
            # Atualizar barra de progresso
            pbar.set_postfix(
                BitAcc=f"{np.mean(bit_acc_scores[-current_batch_size:]):.1f}%",
                MsgAcc=f"{np.mean(message_acc_scores[-current_batch_size:]):.1f}%",
                SSIM=f"{np.mean(ssim_scores[-current_batch_size:]):.4f}"
            )
        
        pbar.close()
    
    print(f"\n{'='*60}")
    print("Teste concluído. Processando resultados finais...")
    print(f"{'='*60}\n")
    
    # --- Calcular Métricas Finais ---
    
    # Gerador
    avg_psnr = np.mean(psnr_scores)
    std_psnr = np.std(psnr_scores)
    
    avg_ssim = np.mean(ssim_scores)
    std_ssim = np.std(ssim_scores)
    
    avg_bit_acc = np.mean(bit_acc_scores)
    std_bit_acc = np.std(bit_acc_scores)
    
    avg_msg_acc = np.mean(message_acc_scores)
    
    # Discriminador
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, 
                                   target_names=['Real', 'Stego'], 
                                   output_dict=True, zero_division=0)
    
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    
    disc_accuracy = report['accuracy'] * 100
    disc_precision = report['Stego']['precision'] * 100
    disc_recall = report['Stego']['recall'] * 100
    disc_f1 = report['Stego']['f1-score'] * 100
    
    # --- Salvar Métricas em JSON ---
    metrics_dict = {
        'generator': {
            'psnr_mean': float(avg_psnr),
            'psnr_std': float(std_psnr),
            'ssim_mean': float(avg_ssim),
            'ssim_std': float(std_ssim),
            'bit_accuracy_mean': float(avg_bit_acc),
            'bit_accuracy_std': float(std_bit_acc),
            'message_accuracy': float(avg_msg_acc)
        },
        'discriminator': {
            'accuracy': float(disc_accuracy),
            'precision': float(disc_precision),
            'recall': float(disc_recall),
            'f1_score': float(disc_f1),
            'auc': float(roc_auc),
            'confusion_matrix': cm.tolist()
        },
        'config': {
            'discriminator_type': args.disc,
            'model_suffix': args.model_suffix,
            'total_images': global_image_count
        }
    }
    
    with open(metrics_json, 'w') as f:
        json.dump(metrics_dict, f, indent=4)
    
    # --- Escrever Sumário ---
    summary_text = f"""
{'='*60}
RELATÓRIO FINAL DE TESTE
{'='*60}

Configuração:
  Discriminador: {args.disc.upper()}
  Modelo: {args.model_suffix}
  Total de Imagens: {global_image_count}

{'='*60}
GERADOR - Qualidade de Imagem
{'='*60}
  PSNR (dB):  {avg_psnr:.4f} ± {std_psnr:.4f}
  SSIM:       {avg_ssim:.6f} ± {std_ssim:.6f}

{'='*60}
GERADOR - Recuperação de Dados
{'='*60}
  Bit Accuracy:     {avg_bit_acc:.2f}% ± {std_bit_acc:.2f}%
  Message Accuracy: {avg_msg_acc:.2f}%

{'='*60}
DISCRIMINADOR - Detecção de Esteganografia
{'='*60}
  Acurácia:  {disc_accuracy:.2f}%
  Precisão:  {disc_precision:.2f}%
  Recall:    {disc_recall:.2f}%
  F1-Score:  {disc_f1:.2f}%
  AUC-ROC:   {roc_auc:.4f}

Matriz de Confusão:
            Predito Real  Predito Stego
Real             {cm[0,0]:6d}        {cm[0,1]:6d}
Stego            {cm[1,0]:6d}        {cm[1,1]:6d}

{'='*60}
Arquivos Gerados
{'='*60}
  Métricas (JSON):       {metrics_json}
  Resultados (CSV):      {csv_file}
  Comparações Visuais:   {comparisons_dir}/
  Gráficos:              {graphs_dir}/

"""
    
    print(summary_text)
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(summary_text)
    
    # --- Gerar Visualizações ---
    print("Gerando visualizações...")
    
    # 1. Matriz de Confusão
    cm_path = os.path.join(graphs_dir, f'confusion_matrix_{args.disc}_{args.model_suffix}.png')
    plot_confusion_matrix(cm, cm_path, args.disc, args.model_suffix)
    print(f"✓ Matriz de Confusão: {cm_path}")
    
    # 2. Curva ROC
    roc_path = os.path.join(graphs_dir, f'roc_curve_{args.disc}_{args.model_suffix}.png')
    plot_roc_curve(fpr, tpr, roc_auc, roc_path, args.disc, args.model_suffix)
    print(f"✓ Curva ROC: {roc_path}")
    
    # 3. Histogramas
    save_histogram(psnr_scores, 'Distribuição de PSNR', 'PSNR (dB)', 
                  os.path.join(graphs_dir, f'hist_psnr_{args.disc}_{args.model_suffix}.png'))
    print(f"✓ Histograma PSNR")
    
    save_histogram(ssim_scores, 'Distribuição de SSIM', 'SSIM', 
                  os.path.join(graphs_dir, f'hist_ssim_{args.disc}_{args.model_suffix}.png'),
                  range_min=0.8, range_max=1.0)
    print(f"✓ Histograma SSIM")
    
    save_histogram(bit_acc_scores, 'Distribuição de Bit Accuracy', 'Bit Accuracy (%)', 
                  os.path.join(graphs_dir, f'hist_bit_acc_{args.disc}_{args.model_suffix}.png'),
                  range_min=80, range_max=100)
    print(f"✓ Histograma Bit Accuracy")
    
    save_histogram(message_acc_scores, 'Distribuição de Message Accuracy', 'Message Accuracy (%)', 
                  os.path.join(graphs_dir, f'hist_msg_acc_{args.disc}_{args.model_suffix}.png'),
                  range_min=0, range_max=100)
    print(f"✓ Histograma Message Accuracy")
    
    print(f"\n{'='*60}")
    print("✅ TESTE COMPLETO!")
    print(f"{'='*60}")
    print(f"Todos os resultados salvos em: {args.results_dir}")
    print(f"{'='*60}\n")