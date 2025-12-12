import torch
import numpy as np
from config import MAX_MESSAGE_LENGTH, BITS_PER_CHAR, MAX_BITS


class ValidationError(Exception):
    """Exceção customizada para erros de validação."""
    pass


def text_to_bits(text):
    """Converte texto ASCII/UTF-8 para uma lista de bits (0s e 1s).
    
    Args:
        text (str): Texto a ser convertido.
        
    Returns:
        list: Lista de bits (0s e 1s).
        
    Raises:
        ValidationError: Se o texto não for uma string.
    """
    if not isinstance(text, str):
        raise ValidationError(f"Esperado tipo 'str', recebeu '{type(text).__name__}'")
    
    bits = []
    for char in text:
        try:
            ascii_val = ord(char)
            if ascii_val > 255: # Garante que cabe em 8 bits
                ascii_val = ord('?') # Substitui por '?' se for inválido
        except TypeError:
            ascii_val = ord('?') # Substitui por '?' se for inválido
            
        # Converte o valor ASCII para 8 bits
        char_bits = bin(ascii_val)[2:].zfill(BITS_PER_CHAR)
        bits.extend([int(b) for b in char_bits])
    return bits

def bits_to_text(bits):
    """Converte uma lista de bits (0s e 1s) de volta para texto ASCII."""
    text = ""
    for i in range(0, len(bits) - (len(bits) % BITS_PER_CHAR), BITS_PER_CHAR):
        byte_bits = bits[i:i+BITS_PER_CHAR]
        
        # Ignora se o bloco não tiver 8 bits (importante para o final)
        if len(byte_bits) != BITS_PER_CHAR:
            continue
            
        byte_str = "".join(map(str, byte_bits))
        
        try:
            ascii_val = int(byte_str, 2)
        except ValueError:
            ascii_val = 0 # Considera como NULO se inválido
        
        if ascii_val == 0:
            # Caractere NULO, para a decodificação (fim da string)
            break
        
        try:
            text += chr(ascii_val)
        except ValueError:
             # Se o código ASCII for inválido, ignora
             text += '?' # Ou pode escolher ignorar com 'continue'
             
    return text

def text_to_tensor(text, device):
    """
    Converte uma string de texto num tensor 1D de bits (tamanho MAX_BITS=800)
    e retorna o tensor e o comprimento real da mensagem em bits.
    
    Args:
        text (str): Texto a ser codificado.
        device (torch.device): Device onde criar o tensor (CPU ou CUDA).
        
    Returns:
        tuple: (tensor, message_length)
            - tensor: Tensor 1D de tamanho MAX_BITS com bits da mensagem
            - message_length: Comprimento real da mensagem em bits
            
    Raises:
        ValidationError: Se texto não for string ou for muito longo.
    """
    if not isinstance(text, str):
        raise ValidationError(f"Esperado tipo 'str', recebeu '{type(text).__name__}'")
    
    if len(text) > MAX_MESSAGE_LENGTH:
        raise ValidationError(
            f"Mensagem muito longa: {len(text)} caracteres > {MAX_MESSAGE_LENGTH} permitidos"
        )
    
    text_bits = text_to_bits(text)
    
    # Trunca se for maior que MAX_BITS (precaução extra)
    if len(text_bits) > MAX_BITS:
        text_bits = text_bits[:MAX_BITS]
        
    message_length = len(text_bits) # 2. O comprimento real em bits
    
    # Cria um tensor de zeros (padding de bits)
    tensor = torch.zeros(MAX_BITS, dtype=torch.float32, device=device)
    
    # Preenche o tensor com os bits da mensagem
    if message_length > 0:
        tensor[:message_length] = torch.tensor(text_bits, dtype=torch.float32)
        
    # 3. Retorna o tensor E o comprimento
    return tensor, message_length

def tensor_to_text(tensor):
    """
    Converte um tensor 1D (ou 2D/3D/4D achatado) de bits (logits 0-1)
    de volta para uma string.
    
    Args:
        tensor (torch.Tensor): Tensor com bits codificados (valores entre 0 e 1).
        
    Returns:
        str: Texto decodificado.
        
    Raises:
        ValidationError: Se o tensor não for um torch.Tensor válido.
    """
    if not isinstance(tensor, torch.Tensor):
        raise ValidationError(f"Esperado 'torch.Tensor', recebeu '{type(tensor).__name__}'")
    
    # Garante que o tensor está na CPU e é 1D
    tensor = tensor.cpu().view(-1)
    
    # Arredonda os logits (saídas da Sigmoid) para 0 ou 1
    bits = (tensor > 0.5).int().numpy()
    
    # Pega apenas os bits relevantes (até MAX_BITS=800)
    bits = bits[:MAX_BITS]
    
    return bits_to_text(bits)

def ask_yes_no(prompt):
    """Faz uma pergunta de sim/não ao utilizador."""
    while True:
        # Garante que a vírgula (se houver) seja removida e formata a pergunta
        prompt_formatada = prompt.strip().rstrip(',')
        response = input(f"{prompt_formatada}? (s/n): ").strip().lower()
        if response in ['s', 'sim']:
            return True
        if response in ['n', 'nao', 'não']:
            return False
        print("Resposta inválida. Por favor, digite 's' ou 'n'.")