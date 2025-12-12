import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import argparse 
from tqdm import tqdm 

from config import (
    DATA_DEPTH, HIDDEN_SIZE, IMAGE_SIZE, BATCH_SIZE,
    LEARNING_RATE, BETA, GAMMA, DEVICE
)
from dataloader import DataLoader
from encoder import BasicEncoder
from decoder import BasicDecoder
from discriminator import BasicDiscriminator
from model import Srnet
from utils import text_to_tensor, MAX_BITS 
from pytorch_msssim import SSIM 

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Treinamento da GAN Esteganográfica')
    parser.add_argument('--disc', type=str, default='basic', choices=['basic', 'srnet'],
                        help='Tipo de discriminador a ser usado: "basic" (PatchGAN) ou "srnet" (Classificador).')
    
    parser.add_argument('--train-dir', type=str, required=True,
                        help='Caminho para a pasta de imagens de treino.')
    parser.add_argument('--val-dir', type=str, required=True,
                        help='Caminho para a pasta de imagens de validação.')
    parser.add_argument('--save-dir', type=str, default='checkpoints',
                        help='Pasta para salvar checkpoints e modelos finais.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Número de épocas de treino.')
    
    args = parser.parse_args()
    
    DATA_PATH = args.train_dir
    VAL_PATH = args.val_dir
    SAVE_PATH = args.save_dir
    NUM_EPOCHS = args.epochs
    
    FRASES_PATH = "frases_geradas.txt"
    
    PADDING_WEIGHT = 0.01 
    
    device = DEVICE
    print(f"Usando o dispositivo: {device}")
    
    os.makedirs(SAVE_PATH, exist_ok=True)

    encoder = BasicEncoder(profundidade_dados=DATA_DEPTH, tamanho_dados_escondidos=HIDDEN_SIZE).to(device)
    decoder = BasicDecoder(profundidade_dados=DATA_DEPTH, tamanho_dados_escondidos=HIDDEN_SIZE).to(device)
    
    if args.disc == 'srnet':
        print("Usando o discriminador: SRNet")
        discriminator = Srnet().to(device)
        checkpoint_path = os.path.join(SAVE_PATH, 'checkpoint_gan_srnet_crossentropy_ssim.pth')
    else:
        print("Usando o discriminador: BasicDiscriminator (PatchGAN)")
        discriminator = BasicDiscriminator(hidden_size=HIDDEN_SIZE).to(device) 
        checkpoint_path = os.path.join(SAVE_PATH, 'checkpoint_gan_basic_crossentropy_ssim.pth')
    
    
    criterio_dados = nn.BCELoss(reduction='none') 
    
    criterio_imagem = SSIM(data_range=1.0, size_average=True, channel=3)  
    
    criterio_adversarial = nn.CrossEntropyLoss() 

    optimizer_E_D = optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()), 
        lr=LEARNING_RATE
    )
    optimizer_Disc = optim.Adam(
        discriminator.parameters(), 
        lr=LEARNING_RATE
    )
        
    start_epoch = 0
    best_val_loss_geral = float('inf') 

    if os.path.exists(checkpoint_path):
        print("Checkpoint encontrado! Carregando progresso...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        try:
            encoder.load_state_dict(checkpoint['encoder_state_dict'])
            decoder.load_state_dict(checkpoint['decoder_state_dict'])
            discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            optimizer_E_D.load_state_dict(checkpoint['optimizer_E_D_state_dict'])
            optimizer_Disc.load_state_dict(checkpoint['optimizer_Disc_state_dict'])
            start_epoch = checkpoint['epoch']
            best_val_loss_geral = checkpoint.get('best_val_loss', float('inf')) 
            print(f"Progresso carregado. Reiniciando da época {start_epoch}. Melhor Loss GERAL (E/D): {best_val_loss_geral:.6f}")
        except KeyError as e:
            print(f"Erro ao carregar o checkpoint. Algumas chaves estão faltando: {e}. Começando do zero.")
            start_epoch = 0
            best_val_loss_geral = float('inf')
    else:
        print("Nenhum checkpoint encontrado. Iniciando treinamento do zero.")

    print("Inicializando o DataLoader de Treino...")
    dataloader = DataLoader(DATA_PATH, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE)
    print("DataLoader de Treino pronto.")
    print("Inicializando o DataLoader de Validação...")
    try:
        val_dataloader = DataLoader(VAL_PATH, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE, shuffle=False)
        print("DataLoader de Validação pronto.")
    except Exception as e:
        print(f"ERRO: Não foi possível carregar o DataLoader de Validação. Verifique se a pasta '{VAL_PATH}' existe.")
        val_dataloader = None
    print("Carregando frases para o treinamento...")
    try:
        with open(FRASES_PATH, 'r', encoding='utf-8') as f:
            training_phrases = [line.strip() for line in f.readlines()]
        print(f"{len(training_phrases)} frases carregadas.")
    except FileNotFoundError:
        print(f"ERRO: Ficheiro de frases '{FRASES_PATH}' não encontrado. Execute frases.py primeiro.")
        exit()

    print(f"Iniciando o treinamento (GAN com Disc: {args.disc}, Loss: CrossEntropy, ImgLoss: SSIM)...")
    print(f"Diretório de Treino: {DATA_PATH}")
    print(f"Diretório de Validação: {VAL_PATH}")
    print(f"Total de Épocas: {NUM_EPOCHS}")
    
    for epoch in range(start_epoch, NUM_EPOCHS):
        encoder.train()
        decoder.train()
        discriminator.train()
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [TREINO]", unit="batch")

        for cover_image_batch, _ in progress_bar:
            cover_image_batch = cover_image_batch.to(device)
            current_batch_size = cover_image_batch.size(0)

            batch_phrases_indices = np.random.choice(len(training_phrases), current_batch_size, replace=False)
            batch_phrases = [training_phrases[i] for i in batch_phrases_indices]
            results = [text_to_tensor(p, device) for p in batch_phrases]
            secret_tensors_flat = torch.stack([r[0] for r in results]) 
            message_lengths = [r[1] for r in results] 
            secret_data_tensor = torch.zeros(current_batch_size, DATA_DEPTH, IMAGE_SIZE, IMAGE_SIZE, device=device)
            secret_data_tensor.view(current_batch_size, -1)[:, :MAX_BITS] = secret_tensors_flat
            loss_weight_mask = torch.full((current_batch_size, MAX_BITS), PADDING_WEIGHT, device=device)
            for i, length in enumerate(message_lengths):
                if length > 0:
                    loss_weight_mask[i, :length] = 1.0 
            loss_weight_mask_4d = torch.zeros_like(secret_data_tensor)
            loss_weight_mask_4d.view(current_batch_size, -1)[:, :MAX_BITS] = loss_weight_mask

            # --- Geração da Imagem ---
            stego_image = encoder(cover_image_batch, secret_data_tensor)

            # --- Etapa 1: Treinar o Discriminador ---
            optimizer_Disc.zero_grad()
            pred_real = discriminator(cover_image_batch)
            pred_fake = discriminator(stego_image.detach()) 
            
            labels_real = torch.full((pred_real.size(0), *pred_real.shape[2:]), 1, 
                                     device=device, dtype=torch.long)
            labels_fake = torch.full((pred_fake.size(0), *pred_fake.shape[2:]), 0, 
                                     device=device, dtype=torch.long)
            
            loss_disc_real = criterio_adversarial(pred_real, labels_real)
            loss_disc_fake = criterio_adversarial(pred_fake, labels_fake)
            loss_disc_total = (loss_disc_real + loss_disc_fake) / 2
            loss_disc_total.backward()
            optimizer_Disc.step()

            # --- Etapa 2: Treinar o Gerador (Encoder) e o Decoder ---
            optimizer_E_D.zero_grad()
            decoded_data_tensor = decoder(stego_image) 
            
            raw_loss_dados = criterio_dados(decoded_data_tensor, secret_data_tensor)
            weighted_loss_dados = raw_loss_dados * loss_weight_mask_4d
            loss_dados = weighted_loss_dados.sum() / (loss_weight_mask_4d.sum() + 1e-8)
            
            loss_imagem = 1.0 - criterio_imagem(stego_image, cover_image_batch)
            
            pred_fake_G = discriminator(stego_image)
            labels_real_G = torch.full((pred_fake_G.size(0), *pred_fake_G.shape[2:]), 1, 
                                       device=device, dtype=torch.long)
            loss_G_adversarial = criterio_adversarial(pred_fake_G, labels_real_G) 
            
            loss_E_D_total = loss_dados + (BETA * loss_imagem) + (GAMMA * loss_G_adversarial)
            
            loss_E_D_total.backward()
            optimizer_E_D.step()
            
            progress_bar.set_postfix(
                Dados=f"{loss_dados.item():.4f}", 
                Img=f"{loss_imagem.item():.4f}", # Agora é (1 - SSIM)
                Adv_G=f"{loss_G_adversarial.item():.4f}",
                Disc=f"{loss_disc_total.item():.4f}"
            )
        
        # --- Ciclo de Validação ---
        if val_dataloader:
            encoder.eval()
            decoder.eval()
            discriminator.eval()
            
            total_val_loss_E_D = 0
            total_val_loss_dados = 0
            total_val_loss_imagem = 0
            total_val_loss_G_adv = 0
            total_val_loss_disc = 0
            
            val_progress_bar = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [VALIDAÇÃO]", unit="batch")
            
            with torch.no_grad():
                for val_cover_batch, _ in val_progress_bar:
                    val_cover_batch = val_cover_batch.to(device)
                    current_batch_size = val_cover_batch.size(0)

                    batch_phrases_indices = np.random.choice(len(training_phrases), current_batch_size, replace=False)
                    batch_phrases = [training_phrases[i] for i in batch_phrases_indices]
                    results = [text_to_tensor(p, device) for p in batch_phrases]
                    secret_tensors_flat = torch.stack([r[0] for r in results])
                    message_lengths = [r[1] for r in results]
                    secret_data_tensor = torch.zeros(current_batch_size, DATA_DEPTH, IMAGE_SIZE, IMAGE_SIZE, device=device)
                    secret_data_tensor.view(current_batch_size, -1)[:, :MAX_BITS] = secret_tensors_flat
                    loss_weight_mask = torch.full((current_batch_size, MAX_BITS), PADDING_WEIGHT, device=device)
                    for i, length in enumerate(message_lengths):
                        if length > 0:
                            loss_weight_mask[i, :length] = 1.0
                    loss_weight_mask_4d = torch.zeros_like(secret_data_tensor)
                    loss_weight_mask_4d.view(current_batch_size, -1)[:, :MAX_BITS] = loss_weight_mask

                    # --- Cálculos das Perdas (Validação) ---
                    stego_image = encoder(val_cover_batch, secret_data_tensor)
                    decoded_data_tensor = decoder(stego_image)

                    raw_loss_dados = criterio_dados(decoded_data_tensor, secret_data_tensor)
                    weighted_loss_dados = raw_loss_dados * loss_weight_mask_4d
                    loss_dados = weighted_loss_dados.sum() / (loss_weight_mask_4d.sum() + 1e-8)

                    loss_imagem = 1.0 - criterio_imagem(stego_image, val_cover_batch)
                    
                    pred_real = discriminator(val_cover_batch)
                    pred_fake_G = discriminator(stego_image)
                    labels_real = torch.full((pred_real.size(0), *pred_real.shape[2:]), 1, 
                                             device=device, dtype=torch.long)
                    labels_fake = torch.full((pred_fake_G.size(0), *pred_fake_G.shape[2:]), 0, 
                                             device=device, dtype=torch.long)
                    
                    loss_G_adversarial = criterio_adversarial(pred_fake_G, labels_real)
                    loss_disc_total = (criterio_adversarial(pred_real, labels_real) + criterio_adversarial(pred_fake_G, labels_fake)) / 2
                    
                    loss_E_D_total = loss_dados + (BETA * loss_imagem) + (GAMMA * loss_G_adversarial)
                    
                    total_val_loss_E_D += loss_E_D_total.item()
                    total_val_loss_dados += loss_dados.item()
                    total_val_loss_imagem += loss_imagem.item()
                    total_val_loss_G_adv += loss_G_adversarial.item()
                    total_val_loss_disc += loss_disc_total.item()

            avg_val_loss_E_D = total_val_loss_E_D / len(val_dataloader)
            avg_val_loss_dados = total_val_loss_dados / len(val_dataloader)
            avg_val_loss_imagem = total_val_loss_imagem / len(val_dataloader)
            avg_val_loss_G_adv = total_val_loss_G_adv / len(val_dataloader)
            avg_val_loss_disc = total_val_loss_disc / len(val_dataloader)
            print(f'\nEpoch [{epoch+1}/{NUM_EPOCHS}] Resumo da Validação:')
            print(f'  [Loss Dados]: {avg_val_loss_dados:<10.6f} | [Loss Imagem (1-SSIM)]: {avg_val_loss_imagem:<10.6f} | [Loss Adv. G]: {avg_val_loss_G_adv:<10.6f} | [Loss Disc]: {avg_val_loss_disc:<10.6f}')
            print(f'  -> Loss GERAL (E/D) da Época: {avg_val_loss_E_D:.6f}')

            if avg_val_loss_E_D < best_val_loss_geral:
                print(f"  -> Nova melhor Loss GERAL (E/D)! ({best_val_loss_geral:.6f} -> {avg_val_loss_E_D:.6f}). Salvando modelos...")
                best_val_loss_geral = avg_val_loss_E_D
                torch.save(encoder.state_dict(), os.path.join(SAVE_PATH, f'encoder_best_{args.disc}_ce_ssim.pth'))
                torch.save(decoder.state_dict(), os.path.join(SAVE_PATH, f'decoder_best_{args.disc}_ce_ssim.pth'))
                torch.save(discriminator.state_dict(), os.path.join(SAVE_PATH, f'discriminator_best_{args.disc}_ce_ssim.pth'))
            print("-" * 80) 
        
        torch.save({
            'epoch': epoch + 1,  
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'optimizer_E_D_state_dict': optimizer_E_D.state_dict(), 
            'optimizer_Disc_state_dict': optimizer_Disc.state_dict(), 
            'loss_E_D': avg_val_loss_E_D if val_dataloader else 0,
            'loss_Disc': avg_val_loss_disc if val_dataloader else 0,
            'best_val_loss': best_val_loss_geral,
            'discriminator_type': args.disc 
        }, checkpoint_path) # 'checkpoint_path' já tem o nome correto

    print("Treinamento concluído!")

    encoder_save_path = os.path.join(SAVE_PATH, f'encoder_gan_{args.disc}_ce_ssim.pth')
    decoder_save_path = os.path.join(SAVE_PATH, f'decoder_gan_{args.disc}_ce_ssim.pth')
    discriminator_save_path = os.path.join(SAVE_PATH, f'discriminator_gan_{args.disc}_ce_ssim.pth')
    
    torch.save(encoder.state_dict(), encoder_save_path)
    torch.save(decoder.state_dict(), decoder_save_path)
    torch.save(discriminator.state_dict(), discriminator_save_path)
    
    print(f"Modelos finais (última época) salvos em: '{encoder_save_path}', '{decoder_save_path}' e '{discriminator_save_path}'")
    print(f"Melhores modelos (baseado em validação GERAL) salvos como '..._best_{args.disc}_ce_ssim.pth' em '{SAVE_PATH}'")