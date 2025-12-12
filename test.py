import torch
import os
from PIL import Image
from torchvision import transforms
import argparse 
from pytorch_msssim import ssim 

from config import DATA_DEPTH, HIDDEN_SIZE, IMAGE_SIZE, DEVICE
from encoder import BasicEncoder
from decoder import BasicDecoder
from utils import text_to_tensor, tensor_to_text, MAX_BITS

def calculate_psnr(img1, img2, data_range=1.0):
    """Calcula o PSNR entre duas imagens tensor (formato NCHW)."""
    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same dimensions.")
    
    # MSE é calculado por imagem no batch
    mse = torch.mean((img1 - img2) ** 2, dim=[1, 2, 3])
    
    # Lida com o caso de MSE ser zero (imagens idênticas)
    psnr = torch.where(mse > 0, 
                      10 * torch.log10(data_range**2 / mse), 
                      torch.tensor(float('inf'), device=mse.device))
    
    return torch.mean(psnr)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Codifica e decodifica uma mensagem em uma imagem.')
    parser.add_argument('-i', '--image', type=str, required=True, 
                        help='Caminho para a imagem de cobertura.')
    parser.add_argument('-m', '--message', type=str, required=True, 
                        help='A mensagem secreta para esconder.')
    parser.add_argument('--dataset', type=str, default='coco', choices=['coco', 'div2k'], 
                        help="Dataset usado no treinamento: 'coco' ou 'div2k'. Padrão: 'coco'")
    parser.add_argument('--discriminator', type=str, default='basic', choices=['basic', 'srnet'],
                        help='Tipo de discriminador usado no treinamento: "basic" ou "srnet". Padrão: "basic"')
    parser.add_argument('--version', type=str, default='best', choices=['best', 'gan'], 
                        help="Versão do modelo a carregar: 'best' (melhor validação) ou 'gan' (final). Padrão: 'best'")
    parser.add_argument('--final', action='store_true',
                        help='Usar versão final do dataset (flag).')
    
    args = parser.parse_args()

    # Construir caminho do checkpoint baseado nos argumentos
    DATASET = args.dataset
    DISCRIMINATOR = args.discriminator
    VERSION = args.version
    FINAL = args.final
    CHECKPOINTS_BASE_DIR = "checkpoints"
    
    # Construir nome da subpasta
    # Formato: {discriminator}_{dataset}_final (se --final) 
    # ou {discriminator}_{dataset} (se não --final)
    # Se o diretório não-final não existir, faz fallback para _final automaticamente
    if FINAL:
        checkpoint_subdir = f"{DISCRIMINATOR}_{DATASET}_final"
    else:
        # Tentar primeiro sem "_final", se não existir, usar com "_final" como fallback
        checkpoint_subdir_non_final = f"{DISCRIMINATOR}_{DATASET}"
        checkpoint_subdir_final = f"{DISCRIMINATOR}_{DATASET}_final"
        
        if os.path.exists(os.path.join(CHECKPOINTS_BASE_DIR, checkpoint_subdir_non_final)):
            checkpoint_subdir = checkpoint_subdir_non_final
        elif os.path.exists(os.path.join(CHECKPOINTS_BASE_DIR, checkpoint_subdir_final)):
            checkpoint_subdir = checkpoint_subdir_final
            print(f"Aviso: Diretório não-final não encontrado, usando '{checkpoint_subdir_final}'")
        else:
            # Nenhum dos diretórios existe - fornecer mensagem de erro clara
            print(f"Erro: Nenhum dos diretórios de checkpoint esperados existe:")
            print(f"  - {CHECKPOINTS_BASE_DIR}/{checkpoint_subdir_non_final}")
            print(f"  - {CHECKPOINTS_BASE_DIR}/{checkpoint_subdir_final}")
            print(f"\nVerifique se:")
            print(f"  1. Os checkpoints foram criados para esta combinação")
            print(f"  2. Os argumentos --dataset e --discriminator estão corretos")
            exit(1)
    
    MODEL_PATH = os.path.join(CHECKPOINTS_BASE_DIR, checkpoint_subdir)
    RESULTS_DIR = "resultados_teste"
    
    # Nome dos arquivos de checkpoint
    # Formato: encoder_{version}_{discriminator}_ce_ssim.pth
    LOSS_SUFFIX = 'ce_ssim'
    
    IMAGE_PATH = args.image
    SECRET_MESSAGE = args.message
    
    ENCODER_MODEL = f'encoder_{VERSION}_{DISCRIMINATOR}_{LOSS_SUFFIX}.pth'
    DECODER_MODEL = f'decoder_{VERSION}_{DISCRIMINATOR}_{LOSS_SUFFIX}.pth'

    ORIGINAL_IMAGE_PATH = os.path.join(RESULTS_DIR, "imagem_original_redimensionada.png")
    STEGO_IMAGE_PATH = os.path.join(RESULTS_DIR, "imagem_codificada.png")
    DECODED_IMAGE_PATH = os.path.join(RESULTS_DIR, "imagem_decodificada_visual.png") # Visualização dos dados
    
    device = DEVICE
    print(f"Usando o dispositivo: {device}")

    # Garante que a pasta de resultados existe
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --- Verificação da Mensagem ---
    max_chars = MAX_BITS // 8 
    if len(SECRET_MESSAGE) > max_chars:
        print(f"Erro: A mensagem é muito longa! Máximo de {max_chars} caracteres.")
        print(f"Sua mensagem tem {len(SECRET_MESSAGE)} caracteres.")
        exit()

    # --- Carregar Modelos ---
    print("Carregando modelos...")
    encoder = BasicEncoder(profundidade_dados=DATA_DEPTH, tamanho_dados_escondidos=HIDDEN_SIZE).to(device)
    decoder = BasicDecoder(profundidade_dados=DATA_DEPTH, tamanho_dados_escondidos=HIDDEN_SIZE).to(device)

    encoder_path = os.path.join(MODEL_PATH, ENCODER_MODEL)
    decoder_path = os.path.join(MODEL_PATH, DECODER_MODEL)

    try:
        encoder.load_state_dict(torch.load(encoder_path, map_location=device))
        decoder.load_state_dict(torch.load(decoder_path, map_location=device))
    except FileNotFoundError:
        print(f"Erro: Arquivos de modelo não encontrados.")
        print(f"  Encoder esperado: {encoder_path}")
        print(f"  Decoder esperado: {decoder_path}")
        print(f"\nVerifique se os argumentos estão corretos:")
        print(f"  --dataset: {DATASET}")
        print(f"  --discriminator: {DISCRIMINATOR}")
        print(f"  --version: {VERSION}")
        print(f"  --final: {FINAL}")
        exit()

    encoder.eval()
    decoder.eval()
    print(f"Modelos '{ENCODER_MODEL}' e '{DECODER_MODEL}' carregados.")

    # --- Preparar Imagem ---
    print(f"Carregando imagem de cobertura: {IMAGE_PATH}")
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])
    save_transform = transforms.ToPILImage()
    
    try:
        image = Image.open(IMAGE_PATH).convert('RGB')
    except FileNotFoundError:
        print(f"ERRO: Não foi possível encontrar a imagem em: {IMAGE_PATH}")
        exit()
        
    cover_image = transform(image).unsqueeze(0).to(device)
    
    original_pil = save_transform(cover_image.squeeze(0).cpu())
    original_pil.save(ORIGINAL_IMAGE_PATH)
    print(f"Imagem original (redimensionada) salva em: {ORIGINAL_IMAGE_PATH}")

    # --- Preparar Dados Secretos ---
    print(f"Preparando mensagem secreta: '{SECRET_MESSAGE}'")
    secret_tensor_flat, message_length = text_to_tensor(SECRET_MESSAGE, device)
    
    secret_data_tensor = torch.zeros(1, DATA_DEPTH, IMAGE_SIZE, IMAGE_SIZE, device=device)
    secret_data_tensor.view(1, -1)[:, :MAX_BITS] = secret_tensor_flat 

    # --- Codificação ---
    print("Codificando mensagem na imagem...")
    with torch.no_grad():
        stego_image_tensor = encoder(cover_image, secret_data_tensor)

    # --- Guardar Imagem Codificada ---
    stego_pil = save_transform(stego_image_tensor.squeeze(0).cpu())
    stego_pil.save(STEGO_IMAGE_PATH)
    print(f"Imagem esteganográfica salva em: {STEGO_IMAGE_PATH}")

    # --- Calcular e Imprimir Métricas de Qualidade ---
    with torch.no_grad():
        current_psnr = calculate_psnr(cover_image, stego_image_tensor, data_range=1.0)
        current_ssim = ssim(cover_image, stego_image_tensor, data_range=1.0, size_average=True)
    
    print("\n--- MÉTRICAS DE QUALIDADE (Original vs. Stego) ---")
    print(f"  PSNR (dB):   {current_psnr.item():.4f}")
    print(f"  SSIM:        {current_ssim.item():.4f}")


    # --- Decodificação ---
    print("\n--- TESTE DE RECUPERAÇÃO ---")
    print(f"Lendo imagem esteganográfica '{STEGO_IMAGE_PATH}' para decodificar...")
    try:
        stego_image_loaded_pil = Image.open(STEGO_IMAGE_PATH).convert('RGB')
        stego_tensor_loaded = transform(stego_image_loaded_pil).unsqueeze(0).to(device) 
    except Exception as e:
        print(f"Erro ao carregar a imagem esteganográfica salva: {e}")
        exit()

    print("Decodificando mensagem lida do disco...")
    with torch.no_grad():
        decoded_data_disk = decoder(stego_tensor_loaded)
    
    # Salva a visualização dos dados decodificados
    decoded_pil = save_transform(decoded_data_disk.squeeze(0).cpu())
    decoded_pil.save(DECODED_IMAGE_PATH)
    print(f"Imagem decodificada salva em: {DECODED_IMAGE_PATH}")

    # Converte os bits de volta para texto
    try:
        recovered_message = tensor_to_text(decoded_data_disk)
    except Exception as e:
        recovered_message = f"[Erro ao decodificar: {e}]"
        
    print(f"Mensagem Recuperada: '{recovered_message}'")

    # --- Verificação Final ---
    if recovered_message.strip() == SECRET_MESSAGE.strip():
        print("\nSucesso! A mensagem original e a recuperada (do disco) são idênticas.")
    else:
        print("\nFalha! A mensagem recuperada (do disco) é diferente da original.")
        print(f"Original:  '{SECRET_MESSAGE}'")
        print(f"Recuperada: '{recovered_message}'")