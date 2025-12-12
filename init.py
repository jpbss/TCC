import os
import sys
import shutil
import requests
import zipfile
import random
from tqdm import tqdm
import time
import subprocess 
import importlib.metadata 
import re 

try:
    from utils import ask_yes_no
    import separar_dados
    import frases
except ImportError:
    print("Erro: Não foi possível encontrar 'utils.py', 'separar_dados.py' ou 'frases.py'.")
    print("Certifique-se de que todos os arquivos estão na mesma pasta.")
    sys.exit(1)


def check_libraries():
    """Verifica se as bibliotecas do requirements.txt estão instaladas."""
    print("--- Verificando bibliotecas... ---")
    requirements_file = 'requirements.txt'
    
    if not os.path.exists(requirements_file):
        print(f"Aviso: '{requirements_file}' não encontrado. Recomendando instalação.")
        return False # Força a pergunta de instalação

    try:
        with open(requirements_file, 'r') as f:
            lines = f.readlines()
        
        # Regex para extrair o nome do pacote (ex: "torch", "numpy", "pytorch-msssim")
        req_pattern = re.compile(r"^\s*([a-zA-Z0-9_\-]+)")
        
        if not lines:
            print("requirements.txt está vazio.")
            return True 

        missing_packages = []

        for line in lines:
            line = line.strip()
            # Ignora comentários ou linhas vazias
            if not line or line.startswith('#'):
                continue
            
            match = req_pattern.match(line)
            if not match:
                continue # Linha de requerimento mal formatada ou vazia
                
            package_name = match.group(1).strip()
            
            try:
                importlib.metadata.version(package_name)
            except importlib.metadata.PackageNotFoundError:
                print(f"Aviso: Biblioteca '{package_name}' parece não estar instalada.")
                missing_packages.append(package_name)

        if missing_packages:
            return False 
        
        print("Todas as bibliotecas do requirements.txt parecem estar instaladas.")
        return True

    except Exception as e:
        print(f"Aviso: Falha ao verificar bibliotecas ({e}). Recomendando instalação.")
        return False

def install_libraries():
    """Tenta instalar as bibliotecas Python a partir do requirements.txt."""
    
    requirements_file = 'requirements.txt'

    print("\n--- Instalação de Bibliotecas ---")
    print(f"Vou tentar instalar/atualizar as bibliotecas a partir do '{requirements_file}'.")
    print("\nAviso Importante: O PyTorch (torch) e Torchvision estão listados no")
    print("requirements.txt, mas a sua instalação pode falhar se o 'pip' não")
    print("encontrar a versão correta para o seu sistema (ex: CUDA vs CPU).")
    print("Se falhar, instale-os manualmente (https://pytorch.org/) e rode este script novamente.\n")
    
    if not os.path.exists(requirements_file):
        print(f"ERRO: Arquivo '{requirements_file}' não encontrado.")
        sys.exit() 

    print(f"--- Lendo {requirements_file} e instalando... ---")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_file])
        print(f"Bibliotecas instaladas/verificadas com sucesso.")
    except subprocess.CalledProcessError as e:
        print(f"ERRO: Falha ao instalar as bibliotecas do '{requirements_file}'.")
        print("Isto pode ter acontecido por causa do PyTorch (veja o aviso acima).")
        print("Tente instalar o PyTorch manualmente e rode o script de novo.")
        print(f"Erro detalhado: {e}")
        if not ask_yes_no("Tentar continuar mesmo assim?"):
            print("Encerrando o script.")
            sys.exit()
    except FileNotFoundError:
         print(f"ERRO: 'pip' não foi encontrado. Certifique-se de que o Python e o pip estão instalados.")
         print("Encerrando o script.")
         sys.exit()
    
    print("\nInstalação das bibliotecas concluída.")
    time.sleep(2) 

def download_file(url, filename):
    """Baixa um arquivo com barra de progresso (tqdm)."""
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            block_size = 1024 
            
            with tqdm(total=total_size, unit='iB', unit_scale=True, desc=f"Baixando {os.path.basename(filename)}") as pbar:
                with open(filename, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=block_size):
                        pbar.update(len(chunk))
                        f.write(chunk)
    except requests.exceptions.RequestException as e:
        print(f"\nErro ao baixar {url}: {e}")
        print("Por favor, verifique a sua conexão com a internet ou a URL.")
        return False
    return True

def unzip_file(zip_path, extract_to):
    """Extrai um arquivo .zip com uma barra de progresso (tqdm)."""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            file_list = zip_ref.infolist()
            total_files = len(file_list)
            pbar = tqdm(file_list, total=total_files, unit=" arquivos", 
                        desc=f"Extraindo {os.path.basename(zip_path)}")
            for member in pbar:
                zip_ref.extract(member, path=extract_to)
                
        print(f"Extração de {os.path.basename(zip_path)} concluída.")
        return True
    
    except zipfile.BadZipFile:
        print(f"Erro: O arquivo {zip_path} não é um .zip válido ou está corrompido.")
        return False
    except Exception as e:
        print(f"Erro ao extrair {zip_path}: {e}")
        return False

def move_images_and_cleanup(src_folder, dst_folder, valid_extensions=('.png', '.jpg', '.jpeg')):
    """Move arquivos de imagem de src para dst (com tqdm) e remove src."""
    try:
        if not os.path.exists(dst_folder):
            os.makedirs(dst_folder) 
            
        files_to_move = [
            item for item in os.listdir(src_folder)
            if os.path.isfile(os.path.join(src_folder, item)) and item.lower().endswith(valid_extensions)
        ]
        
        pbar = tqdm(files_to_move, 
                    desc=f"Movendo de '{os.path.basename(src_folder)}'", 
                    unit=" arquivos", 
                    leave=True) 

        for item in pbar:
            src_item_path = os.path.join(src_folder, item)
            dst_item_path = os.path.join(dst_folder, item)
            try:
                shutil.move(src_item_path, dst_item_path)
            except (shutil.Error, FileNotFoundError):
                pass 
        
        shutil.rmtree(src_folder)
        print(f"Limpeza de '{src_folder}' concluída.")
    except Exception as e:
        print(f"Erro ao mover/limpar '{src_folder}': {e}")


def choose_checkpoint_directory(base_dir, for_training=False):
    """
    Lista todas as SUBPASTAS dentro de 'base_dir' (ex: 'checkpoints/') 
    e pergunta ao usuário qual delas usar.
    
    Se for_training=True, oferece a opção de criar uma nova subpasta.
    Retorna o caminho completo (ex: 'checkpoints/basic_coco') ou None se cancelar.
    """
    print(f"\n--- Seleção da Pasta de Checkpoints (Dentro de '{base_dir}') ---")
    
    if not os.path.exists(base_dir):
        print(f"Aviso: A pasta base '{base_dir}' não existe. Criando...")
        os.makedirs(base_dir, exist_ok=True)

    try:
        # Lista subpastas dentro da pasta base de checkpoints
        all_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    except Exception as e:
        print(f"Erro ao listar subpastas em '{base_dir}': {e}")
        return None

    if not all_dirs:
        print("Nenhuma pasta de checkpoint existente encontrada.")
    else:
        print("Pastas de checkpoint existentes:")
        for i, dir_name in enumerate(all_dirs):
            print(f" {i+1}. {dir_name}")
    
    if for_training:
        print(f" {len(all_dirs) + 1}. [CRIAR NOVA SUBPASTA]")
    
    choice = -1
    max_choice = len(all_dirs) + 1 if for_training else len(all_dirs)
    
    if max_choice == 0 and not for_training:
        print("Nenhuma pasta de checkpoint disponível para carregar.")
        return None

    while choice < 1 or choice > max_choice:
        try:
            raw_choice = input(f"Escolha a pasta (1-{max_choice}): ").strip()
            if not raw_choice: continue
            choice = int(raw_choice)
        except ValueError:
            print("Por favor, digite um número.")

    if for_training and choice == max_choice:
        # Criar nova pasta
        new_name = ""
        while not new_name:
            new_name = input("Digite o nome da nova subpasta de treino (ex: srnet_coco): ").strip()
        
        chosen_path = os.path.join(base_dir, new_name)
        os.makedirs(chosen_path, exist_ok=True)
        print(f"Pasta '{chosen_path}' criada/selecionada.")
        return chosen_path
    else:
        # Usar pasta existente
        chosen_dir = all_dirs[choice - 1]
        chosen_path = os.path.join(base_dir, chosen_dir)
        print(f"Pasta selecionada: '{chosen_path}'")
        return chosen_path


def run_training_session(datasets_root, checkpoints_base_dir):
    """
    Inicia uma sessão de treinamento. Pergunta ao usuário o dataset,
    o modelo e, em seguida, executa o train.py via subprocess.
    """
    
    # Pergunta qual SUBPASTA de checkpoint usar/criar.
    checkpoints_dir = choose_checkpoint_directory(checkpoints_base_dir, for_training=True)
    if checkpoints_dir is None:
        print("Seleção de pasta cancelada.")
        return
    
    print("\n--- Iniciando Sessão de Treinamento ---")
    
    # 0. Verifica se as frases existem
    if not os.path.exists('frases_geradas.txt'):
        print("ERRO: O arquivo 'frases_geradas.txt' não foi encontrado.")
        print("Por favor, rode a 'Configuração de Primeira Vez' (Opção 1) ou")
        print("a 'Apenas Gerar Frases' (Opção 5) primeiro.")
        return
        
    # 1. Encontrar datasets válidos (que tenham a pasta 'train/')
    if not os.path.exists(datasets_root):
        print(f"ERRO: Pasta '{datasets_root}' não encontrada.")
        print("Por favor, rode a 'Configuração de Primeira Vez' (Opção 1) primeiro.")
        return

    valid_datasets = []
    try:
        for item in os.listdir(datasets_root):
            dataset_path = os.path.join(datasets_root, item)
            train_path = os.path.join(dataset_path, "train")
            if os.path.isdir(dataset_path) and os.path.isdir(train_path):
                valid_datasets.append(item)
    except FileNotFoundError:
        print(f"ERRO: Pasta '{datasets_root}' não encontrada.")
        return

    if not valid_datasets:
        print(f"ERRO: Nenhum dataset foi encontrado em '{datasets_root}' que já esteja separado.")
        print("Por favor, rode a Opção 1 ou 2 para preparar os dados.")
        return

    # 2. Perguntar ao usuário qual dataset usar
    print("\nDatasets disponíveis (com pasta 'train' encontrada):")
    for i, name in enumerate(valid_datasets):
        print(f" {i+1}. {name}")
    
    dataset_choice = -1
    while dataset_choice < 1 or dataset_choice > len(valid_datasets):
        try:
            raw_choice = input(f"Escolha o dataset (1-{len(valid_datasets)}): ").strip()
            if not raw_choice: continue
            dataset_choice = int(raw_choice)
        except ValueError:
            print("Por favor, digite um número.")
    
    chosen_dataset = valid_datasets[dataset_choice - 1]
    train_dir = os.path.join(datasets_root, chosen_dataset, "train")
    val_dir = os.path.join(datasets_root, chosen_dataset, "val")
    
    if not os.path.exists(val_dir):
         print(f"ERRO: A pasta de validação '{val_dir}' não foi encontrada.")
         return
         
    print(f"Dataset selecionado: '{chosen_dataset}'")

    # 3. Perguntar o modelo
    print("\nModelos disponíveis:")
    print(" 1. basic")
    print(" 2. srnet (SRNet)")
    model_choice = ""
    while model_choice not in ['1', '2']:
        model_choice = input("Escolha o modelo (1-2): ").strip()
    
    chosen_model = "basic" if model_choice == "1" else "srnet"
    print(f"Modelo selecionado: '{chosen_model}'")

    # 4. Perguntar épocas
    epochs = "10" 
    try:
        raw_epochs = input("Quantas épocas de treino? (padrão: 10): ").strip()
        if raw_epochs and int(raw_epochs) > 0:
            epochs = raw_epochs
    except ValueError:
        print("Entrada inválida, usando padrão de 10 épocas.")
        pass

    # 5. Construir e executar o comando
    command = [
        sys.executable,
        "train.py",
        "--disc", chosen_model,
        "--train-dir", train_dir,
        "--val-dir", val_dir,
        "--save-dir", checkpoints_dir, 
        "--epochs", epochs
    ]
    
    print("\n--- Executando o Treinamento ---")
    print(f"Comando: {' '.join(command)}")
    print("A saída do treinamento aparecerá abaixo. (Pressione Ctrl+C para parar)\n")
    
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nERRO: O treinamento falhou com o código de saída {e.returncode}.")
    except KeyboardInterrupt:
        print("\nTreinamento interrompido pelo usuário.")
    except FileNotFoundError:
        print("\nERRO: 'train.py' não encontrado.")

def run_testing_session(datasets_root, checkpoints_base_dir, results_dir):
    """
    Inicia uma sessão de teste em lote (test_batch.py).
    Permite escolher arquitetura, checkpoint e dataset de teste.
    """
    
    checkpoints_dir = choose_checkpoint_directory(checkpoints_base_dir, for_training=False)
    if checkpoints_dir is None:
        print("Seleção de pasta cancelada.")
        return
    
    print("\n--- Iniciando Sessão de Teste em Lote ---")

    # 1. Escolher Arquitetura
    print("\nArquiteturas disponíveis:")
    print(" 1. basic")
    print(" 2. srnet (SRNet)")
    model_choice = ""
    while model_choice not in ['1', '2']:
        model_choice = input("Escolha a arquitetura do modelo (1-2): ").strip()
    
    chosen_model = "basic" if model_choice == "1" else "srnet"
    print(f"Arquitetura selecionada: '{chosen_model}'")

    # 2. Pergunta o SUFIXO do modelo
    print("\nQual tipo de modelo (sufixo) deseja carregar?")
    print(" 1. best (melhor modelo com base na validação)")
    print(" 2. gan (modelo final, da última época)")
    suffix_choice = ""
    while suffix_choice not in ['1', '2']:
        suffix_choice = input("Escolha o sufixo do modelo (1-2): ").strip()
    
    chosen_suffix = "best" if suffix_choice == "1" else "gan"
    print(f"Sufixo selecionado: '{chosen_suffix}'")


    # 3. Escolher Dataset de Teste
    if not os.path.exists(datasets_root):
        print(f"ERRO: Pasta '{datasets_root}' não encontrada.")
        return

    valid_test_sets = []
    try:
        for item in os.listdir(datasets_root):
            dataset_path = os.path.join(datasets_root, item)
            test_path = os.path.join(dataset_path, "test")
            if os.path.isdir(dataset_path) and os.path.isdir(test_path):
                valid_test_sets.append(item)
    except FileNotFoundError:
        print(f"ERRO: Pasta '{datasets_root}' não encontrada.")
        return

    if not valid_test_sets:
        print(f"ERRO: Nenhum dataset foi encontrado em '{datasets_root}' com uma pasta 'test/'.")
        return

    print("\nDatasets de teste disponíveis:")
    for i, name in enumerate(valid_test_sets):
        print(f" {i+1}. {name}")
    
    dataset_choice = -1
    while dataset_choice < 1 or dataset_choice > len(valid_test_sets):
        try:
            raw_choice = input(f"Escolha o dataset de teste (1-{len(valid_test_sets)}): ").strip()
            if not raw_choice: continue
            dataset_choice = int(raw_choice)
        except ValueError:
            print("Por favor, digite um número.")
    
    chosen_dataset = valid_test_sets[dataset_choice - 1]
    test_dir = os.path.join(datasets_root, chosen_dataset, "test")
    print(f"Dataset de teste selecionado: '{chosen_dataset}'")
    
    # 4. Perguntar o número de imagens para testar 
    # Tenta contar as imagens no diretório de teste
    total_images = 0
    try:
        total_images = len([name for name in os.listdir(test_dir) if os.path.isfile(os.path.join(test_dir, name))])
        print(f"O dataset '{chosen_dataset}' contém {total_images} imagens de teste.")
    except Exception:
        print("Não foi possível contar as imagens no dataset de teste.")

    num_tests = "50" 
    try:
        raw_num_tests = input("Quantas imagens deseja testar? (padrão: 50, '0' para todas): ").strip()
        if raw_num_tests:
            num_tests_int = int(raw_num_tests)
            if num_tests_int == 0:
                num_tests = str(sys.maxsize) 
                print("Testando TODAS as imagens.")
            elif num_tests_int > 0:
                num_tests = raw_num_tests
                if total_images > 0 and num_tests_int > total_images:
                    print(f"Aviso: O número pedido ({num_tests_int}) é maior que o disponível ({total_images}). Testando todas.")
                    num_tests = str(sys.maxsize)
            else:
                print("Entrada inválida, usando padrão de 50.")
        else:
            print("Usando padrão de 50.")
            
    except ValueError:
        print("Entrada inválida, usando padrão de 50 imagens.")
        pass 
    
    # 5. Definir pasta de resultados
    checkpoint_folder_name = os.path.basename(checkpoints_dir)
    output_folder_name = f"Resultados_{chosen_dataset}_com_{checkpoint_folder_name}_{chosen_suffix}"
    output_dir = os.path.join(results_dir, output_folder_name)
    print(f"Os resultados serão salvos em: '{output_dir}'")
    os.makedirs(output_dir, exist_ok=True)

    # 6. Construir e executar o comando
    command = [
        sys.executable,
        "test_batch.py",
        "--disc", chosen_model,
        "--model_path", checkpoints_dir, 
        "--model_suffix", chosen_suffix, # Passa o SUFIXO (best/gan)
        "--image_dir", test_dir,         
        "--results_dir", output_dir,
        "--num_tests", num_tests 
    ]

    print("\n--- Executando Teste em Lote ---")
    print(f"Comando: {' '.join(command)}")
    print("A saída do teste aparecerá abaixo. (Pressione Ctrl+C para parar)\n")

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nERRO: O teste em lote falhou com o código de saída {e.returncode}.")
    except KeyboardInterrupt:
        print("\nTreinamento interrompido pelo usuário.")
    except FileNotFoundError:
        print("\nERRO: 'test_batch.py' não encontrado.")


def main():
    print("--- Script de Inicialização do Projeto ---")
    
    print("\nO que deseja fazer?")
    print(" 1. Configuração de Primeira Vez (Instala, Baixa, Separa e Gera Frases)")
    print(" 2. Apenas Re-separar os Dados (Assume que já baixou)")
    print(" 3. Iniciar Treinamento")
    print(" 4. Iniciar Teste em Lote")
    print(" 5. Apenas Gerar Frases (frases_geradas.txt)")
    print(" 6. Sair")
    
    choice = ""
    while choice not in ['1', '2', '3', '4', '5', '6']:
        choice = input("Escolha uma opção (1-6): ").strip()
    
    if choice == '6':
        print("Encerrando o script.")
        sys.exit()

    datasets_root = 'datasets'
    
    datasets_para_baixar = {
        'div2k': {
            'target_folder': 'div2k', 
            'parts': [ 
                {
                    'url': 'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip',
                    'zip_name': 'DIV2K_train_HR.zip',
                    'folder_in_zip': 'DIV2K_train_HR' 
                },
                {
                    'url': 'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip',
                    'zip_name': 'DIV2K_valid_HR.zip',
                    'folder_in_zip': 'DIV2K_valid_HR' 
                }
            ],
            'cleanup_method': 'move'
        },
        'coco': {
            'target_folder': 'coco',
            'parts': [
                {
                    'url': 'http://images.cocodataset.org/zips/unlabeled2017.zip',
                    'zip_name': 'coco_unlabeled2017.zip',
                    'folder_in_zip': 'unlabeled2017' 
                }
            ],
            'cleanup_method': 'rename'
        }
    }
    
    pastas_necessarias = ['checkpoints', 'datasets', 'resultados_teste', 'resultados_batch']
    checkpoints_base_dir = 'checkpoints' 
    results_batch_dir = 'resultados_batch'

    if choice == '1':
        
        libraries_ok = check_libraries()
        if not libraries_ok:
            if ask_yes_no("Algumas bibliotecas parecem estar faltando. Posso tentar instalá-las (via requirements.txt)"):
                install_libraries()
            else:
                print("Instalação de bibliotecas pulada.")
                time.sleep(1)
        else:
            print("Bibliotecas já estão OK. Pulando instalação.\n")
        
        all_folders_exist = all(os.path.exists(p) for p in pastas_necessarias)
        
        if not all_folders_exist:
            if not ask_yes_no("Algumas pastas do projeto estão faltando. Posso criá-las"):
                print("Infelizmente não poderemos continuar. O script será encerrado.")
                sys.exit()

            print("\nCriando estruturas de pastas...")
            for pasta in pastas_necessarias:
                os.makedirs(pasta, exist_ok=True) 
                print(f"- Pasta '{pasta}' criada/verificada.")
            print("Estrutura de pastas base criada com sucesso.\n")
        else:
            print("\nEstrutura de pastas já está OK. Pulando criação.\n")


        print("\n--- Download de Datasets ---")
        print("Vou perguntar quais datasets deseja baixar/instalar.")

        for key, data in datasets_para_baixar.items():
            dataset_final_path = os.path.join(datasets_root, data['target_folder'])
            
            if os.path.exists(dataset_final_path):
                print(f"\nO dataset '{key}' já existe em '{dataset_final_path}'.")
                if not ask_yes_no("Deseja verificar/baixar novamente? (Não recomendado se já estiver completo)"):
                    print(f"Dataset '{key}' pulado.")
                    continue 
            
            if not ask_yes_no(f"Deseja baixar e instalar o dataset '{key}'?"):
                print(f"Download do '{key}' pulado pelo utilizador.")
                continue 
            
            print(f"--- Processando dataset: {key} ---")

            for part in data['parts']:
                zip_path = os.path.join(datasets_root, part['zip_name'])
                temp_extract_folder = os.path.join(datasets_root, part['folder_in_zip'])

                if os.path.exists(zip_path):
                    print(f"Arquivo zip '{part['zip_name']}' já existe. Pulando download.")
                else:
                    if not download_file(part['url'], zip_path):
                        continue 

                if not unzip_file(zip_path, datasets_root):
                    continue 
                
                print(f"Aguarde, limpando o arquivo zip '{part['zip_name']}' (isto pode demorar)...")
                try:
                    os.remove(zip_path)
                    print(f"Arquivo zip '{part['zip_name']}' limpo com sucesso.")
                except OSError as e:
                    print(f"Aviso: Não foi possível remover o arquivo zip '{zip_path}'. {e}")

                
                cleanup_method = data.get('cleanup_method', 'move') 

                if os.path.exists(temp_extract_folder):
                    if cleanup_method == 'rename':
                        if os.path.exists(dataset_final_path):
                             try:
                                 shutil.rmtree(dataset_final_path)
                             except OSError as e:
                                 print(f"Erro: Não foi possível limpar o destino '{dataset_final_path}'. {e}")
                                 continue 
                        try:
                            os.rename(temp_extract_folder, dataset_final_path)
                            print(f"Pasta '{temp_extract_folder}' renomeada para '{dataset_final_path}'.")
                        except OSError as e:
                            print(f"Erro ao renomear pasta: {e}")
                    
                    else:
                        if not os.path.exists(dataset_final_path):
                            print(f"Renomeando (rápido) '{temp_extract_folder}' para '{dataset_final_path}'...")
                            try:
                                os.rename(temp_extract_folder, dataset_final_path)
                            except OSError as e:
                                print(f"Erro ao renomear pasta: {e}")
                        else:
                            print(f"Movendo (lento) de '{temp_extract_folder}' para '{dataset_final_path}'...")
                            move_images_and_cleanup(temp_extract_folder, dataset_final_path)
                else:
                    print(f"Atenção: A pasta extraída '{temp_extract_folder}' não foi encontrada.")
            
            print(f"Dataset '{key}' instalado com sucesso em '{dataset_final_path}'.")
            
        print("\nDownload e instalação dos datasets concluídos.\n")
        time.sleep(1) 
        
        separar_dados.split_datasets(datasets_root, datasets_para_baixar)
        
        print("\n--- Geração de Frases ---")
        try:
            frases.main() 
            print("Geração de frases concluída.")
        except Exception as e:
            print(f"\nERRO: A geração de frases falhou: {e}")
            print("Poderá ter de executar 'python3 frases.py' manualmente.")


    elif choice == '2':
        print("--- Iniciando 'Apenas Re-separar os Dados' ---")
        separar_dados.split_datasets(datasets_root, datasets_para_baixar)

    elif choice == '3':
        run_training_session(datasets_root, checkpoints_base_dir) 

    elif choice == '4':
        run_testing_session(datasets_root, checkpoints_base_dir, results_batch_dir)

    elif choice == '5':
        print("\n--- Iniciando 'Apenas Gerar Frases' ---")
        try:
            frases.main()
        except Exception as e:
            print(f"\nERRO: A geração de frases falhou: {e}")


    print("\n--- Operação Concluída! ---")
    if choice == '1':
        print("O seu ambiente está pronto.")
    elif choice == '2':
        print("A re-separação dos dados foi concluída.")
    elif choice == '3':
        print("Sessão de treinamento finalizada.")
    elif choice == '4':
        print("Sessão de teste em lote finalizada.")
    elif choice == '5':
        print("Geração de frases concluída.")


if __name__ == "__main__":
    main()