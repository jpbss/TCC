import os
import sys
import shutil
import random
import subprocess 
from tqdm import tqdm

try:
    from utils import ask_yes_no
except ImportError:
    print("Erro: Não foi possível encontrar o arquivo 'utils.py'.")
    print("Certifique-se de que 'utils.py' está na mesma pasta.")
    sys.exit(1)

def split_datasets(datasets_root, datasets_para_baixar):
    """
    Itera sobre os datasets, pergunta sobre a divisão
    e move os arquivos.
    """
    print("\n--- Separação de Dados ---")
    
    for dataset_name, data in datasets_para_baixar.items():
        dataset_path = os.path.join(datasets_root, data['target_folder']) 
        
        valid_extensions = ('.png', '.jpg', '.jpeg')
        
        if not os.path.exists(dataset_path):
            print(f"\nAviso: A pasta do dataset '{dataset_path}' não foi encontrada.")
            print(f"Separação pulada para '{dataset_name}'. (Rode a 'Configuração de Primeira Vez' para baixar).")
            continue
            
        print(f"\nA processar dataset: '{dataset_name}' (em '{dataset_path}')")

        if os.path.exists(os.path.join(dataset_path, "train")):
            print(f"O dataset '{dataset_name}' já parece estar separado em pastas train/test/val.")
            if not ask_yes_no("Deseja separá-lo novamente? (Isto apagará a separação atual)"):
                print(f"Separação pulada para '{dataset_name}'.")
                continue
            else:
                print("Restaurando arquivos para a raiz para re-separar...")
                subfolders = ['train', 'test', 'val']
                
                for folder in subfolders:
                    subfolder_path = os.path.join(dataset_path, folder)
                    if not os.path.exists(subfolder_path):
                        continue
                    
                    try:
                        files_to_move = [
                            f for f in os.listdir(subfolder_path) 
                            if f.lower().endswith(valid_extensions)
                        ]
                        
                        # Tenta usar 'mv' do sistema primeiro (muito mais rápido no Drive)
                        if files_to_move:
                            print(f"Restaurando {len(files_to_move)} arquivos de '{folder}' (usando 'mv')...")
                            src_paths = [os.path.join(subfolder_path, f) for f in files_to_move]
                            command = ["mv"] + src_paths + [dataset_path]
                            try:
                                subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                            except (subprocess.CalledProcessError, FileNotFoundError):
                                print("Falha no 'mv', usando método lento (shutil.move) como fallback...")
                                pbar = tqdm(files_to_move, desc=f"Restaurando de '{folder}' (fallback)", unit=" arquivos", leave=False)
                                for img_file in pbar:
                                    src_path = os.path.join(subfolder_path, img_file)
                                    dst_path = os.path.join(dataset_path, img_file)
                                    try:
                                        shutil.move(src_path, dst_path)
                                    except (shutil.Error, FileNotFoundError):
                                        pass 
                        
                        shutil.rmtree(subfolder_path)
                        
                    except OSError as e:
                        print(f"Erro ao limpar a pasta '{folder}': {e}. Pulando.")
                
                print("Arquivos movidos de volta para a raiz. Pronto para a nova separação.")
        
        split_prompt = f"Qual a divisão (treino, teste, validação) para '{dataset_name}'? (padrão: 70,15,15): "
        split_input = input(split_prompt).strip()

        if not split_input:
            splits = [0.7, 0.15, 0.15] # Padrão
        else:
            try:
                parts = [float(x.strip().replace('%','')) for x in split_input.split(',')]
                if len(parts) != 3 or sum(parts) != 100:
                    raise ValueError("A soma deve ser 100 e deve ter 3 partes (ex: 70,15,15).")
                splits = [p / 100.0 for p in parts]
            except Exception as e:
                print(f"Entrada inválida: {e}. Usando padrão 70,15,15.")
                splits = [0.7, 0.15, 0.15]
        
        p_train, p_test, p_val = splits
        print(f"Usando divisão: {p_train*100:.0f}% Treino, {p_test*100:.0f}% Teste, {p_val*100:.0f}% Validação.")

        # Encontra todas as imagens (PNG, JPG, JPEG)
        images = [f for f in os.listdir(dataset_path) 
                  if os.path.isfile(os.path.join(dataset_path, f)) and f.lower().endswith(valid_extensions)]
        
        if not images:
            print(f"Nenhuma imagem encontrada em '{dataset_path}' para separar.")
            continue
        
        #  seed fixo para que o 'shuffle' seja sempre o mesmo
        random.seed(42)
        random.shuffle(images)
        total_images_found = len(images)
        
        print(f"Foram encontradas {total_images_found} imagens em '{dataset_name}'.")
        
        DEFAULT_LIMIT = 20000
        default_num = total_images_found 
        
        if total_images_found > DEFAULT_LIMIT:
            default_num = DEFAULT_LIMIT 
            print(f"O dataset tem > {DEFAULT_LIMIT} imagens. O padrão será usar {DEFAULT_LIMIT}.")

        num_to_use = default_num 
        
        while True:
            try:
                num_prompt = f"Quantas destas {total_images_found} imagens deseja usar? (padrão: {num_to_use}): "
                user_input = input(num_prompt).strip()
                
                if not user_input:
                    break

                num_to_use = int(user_input)
                
                if num_to_use <= 0:
                    print("Por favor, insira um número positivo.")
                elif num_to_use > total_images_found:
                    print(f"Valor inválido. Não pode usar mais do que as {total_images_found} imagens disponíveis.")
                else:
                    break # Número válido
            except ValueError:
                print("Entrada inválida. Por favor, digite um número.")
        
        print(f"Ok, usando {num_to_use} imagens para a separação.")

        # Seleciona o subconjunto de imagens (já aleatórias)
        images_to_use = images[:num_to_use]
        total_images_to_use = len(images_to_use)
        
        # Calcula os índices de separação com base no novo total
        idx_train = int(total_images_to_use * p_train)
        idx_test = int(total_images_to_use * (p_train + p_test))
        
        files_train = images_to_use[:idx_train]
        files_test = images_to_use[idx_train:idx_test]
        files_val = images_to_use[idx_test:] # O resto vai para a validação

        split_folders = {
            "train": files_train,
            "test": files_test,
            "val": files_val
        }

        print("Movendo arquivos para as pastas de divisão...")
        
        for folder_name, file_list in split_folders.items():
            folder_path = os.path.join(dataset_path, folder_name)
            os.makedirs(folder_path, exist_ok=True)
            
            if not file_list:
                print(f"Nenhum arquivo para mover para '{folder_name}'.")
                continue

            # Tenta usar 'mv' do sistema primeiro (muito mais rápido no Drive)
            print(f"Movendo {len(file_list)} arquivos para '{folder_name}' (usando 'mv')...")
            # Constrói os caminhos de origem completos
            src_paths = [os.path.join(dataset_path, f) for f in file_list]
            
            # O comando 'mv' pode falhar se a lista for muito longa 
            command = ["mv"] + src_paths + [folder_path]
            
            try:
                subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print(f"Arquivos movidos para '{folder_name}' com sucesso.")
            
            except (subprocess.CalledProcessError, FileNotFoundError, OSError) as e:
                print(f"Falha no 'mv' para '{folder_name}' (Erro: {e}), usando método lento (shutil.move) como fallback...")
                pbar = tqdm(file_list, desc=f"Movendo para '{folder_name}' (fallback)", unit=" arquivos", leave=False)
                for img_file in pbar:
                    src_path = os.path.join(dataset_path, img_file)
                    dst_path = os.path.join(folder_path, img_file)
                    try:
                        shutil.move(src_path, dst_path)
                    except FileNotFoundError:
                        print(f"Aviso: Arquivo {src_path} não encontrado (pode já ter sido movido).")

        
        print(f"Separação de dados concluída para '{dataset_name}'.\n")

if __name__ == "__main__":
    print("Este script é feito para ser importado pelo 'init.py'.")
    print("Por favor, execute 'python init.py' para começar.")