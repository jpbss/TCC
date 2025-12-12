import random
import string
import os 

ARQUIVO_LEXICO = 'lexico'
ARQUIVO_SAIDA = 'frases_geradas.txt'

try:
    from utils import MAX_MESSAGE_LENGTH
    MAX_LEN = MAX_MESSAGE_LENGTH # 100 caracteres
except ImportError:
    print("Aviso: Não foi possível importar MAX_MESSAGE_LENGTH de utils.py. Usando 100 como padrão.")
    MAX_LEN = 100

MIN_LEN = 3
DEFAULT_NUM_FRASES = 50000 # Alterado para um padrão

def carregar_vocabulario(arquivo):
    """Carrega o arquivo de léxico para a memória."""
    try:
        with open(arquivo, 'r', encoding='utf-8') as f:
            palavras = [linha.strip() for linha in f if linha.strip()]
        if not palavras:
            print(f"Aviso: O arquivo '{arquivo}' está vazio. Usando caracteres aleatórios.")
            return None
        return palavras
    except FileNotFoundError:
        print(f"Erro: Arquivo léxico '{arquivo}' não encontrado. Usando caracteres aleatórios.")
        return None

def gerar_frase_com_palavras(vocabulario, min_len, max_len):
    """
    Gera uma frase com palavras aleatórias do vocabulário,
    com um comprimento aleatório entre min_len e max_len.
    """
    tamanho_alvo = random.randint(min_len, max_len)
    frase = ""
    while len(frase) < tamanho_alvo:
        palavra = random.choice(vocabulario)
        if not frase:
            frase_potencial = palavra
        else:
            frase_potencial = f"{frase} {palavra}"
        
        if len(frase_potencial) > max_len:
            if not frase:
                frase = palavra[:max_len]
            break
        frase = frase_potencial
        if len(frase) >= tamanho_alvo:
            break
    return frase[:max_len]

def gerar_frase_aleatoria_fallback(tamanho):
    """Gera uma string aleatória (método antigo) caso o léxico falhe."""
    CARACTERES_VALIDOS = string.ascii_letters + string.digits + " "
    return ''.join(random.choice(CARACTERES_VALIDOS) for _ in range(tamanho))

def main():
    numero_de_frases = DEFAULT_NUM_FRASES
    try:
        print("\n--- Geração de Frases para Treinamento ---")
        raw_num = input(f"Quantas frases deseja gerar? (padrão: {DEFAULT_NUM_FRASES}): ").strip()
        if raw_num and int(raw_num) > 0:
            numero_de_frases = int(raw_num)
        else:
            print(f"Usando o padrão de {DEFAULT_NUM_FRASES}.")
    except ValueError:
        print(f"Entrada inválida. Usando o padrão de {DEFAULT_NUM_FRASES}.")
    
    vocabulario = carregar_vocabulario(ARQUIVO_LEXICO)
    
    with open(ARQUIVO_SAIDA, 'w', encoding='utf-8') as f:
        print(f"Gerando {numero_de_frases} frases...")
        
        for _ in range(numero_de_frases):
            if vocabulario:
                frase = gerar_frase_com_palavras(vocabulario, MIN_LEN, MAX_LEN)
            else:
                tamanho_aleatorio = random.randint(MIN_LEN, MAX_LEN)
                frase = gerar_frase_aleatoria_fallback(tamanho_aleatorio)
                
            f.write(frase + '\n')
    
    if vocabulario:
        print(f"{numero_de_frases} frases geradas com palavras do léxico e salvas em '{ARQUIVO_SAIDA}'")
        print(f"Os comprimentos variam aleatoriamente de {MIN_LEN} a {MAX_LEN} caracteres.")
    else:
        print(f"{numero_de_frases} frases (fallback) geradas e salvas em '{ARQUIVO_SAIDA}'")

if __name__ == "__main__":
    main()