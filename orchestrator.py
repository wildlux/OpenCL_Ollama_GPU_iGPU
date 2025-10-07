import subprocess
import time
import yaml
import ollama
from ollama.exceptions import OllamaAPIError
import sys
import os # Necessario per controllare il file /etc/hosts

# --- CONFIGURAZIONE GLOBALE ---
# Modelli: SCELTI PER ENTRARE NEI LIMITI VRAM (4GB) senza offload su RAM/CPU
MODEL_SLM_INTEL = "phi3:3.8b" 
MODEL_SLM_NVIDIA = "gemma:2b"

# Perfetto. Hai ragione, la modifica del file /etc/hosts è la soluzione più semplice e robusta per un ambiente Docker locale.
# Ecco lo script Python aggiornato (orchestrator.py), che ora utilizza i nomi INTELoLLAMA e NVIDIAoLLAMA per comunicare con i container, mantenendo inalterata la logica di bilanciamento del carico SLM/LLM e di fall-back che abbiamo definito.
# 🐍 Script Python Aggiornato (orchestrator.py)
# Prima di eseguire lo script, ricordati di aver modificato il tuo file /etc/hosts come segue:
#
# # Mappatura per i servizi Ollama
# 127.0.0.1    INTELoLLAMA
# 127.0.0.1    NVIDIAoLLAMA
HOST_INTEL = "http://INTELoLLAMA:11434"
HOST_NVIDIA = "http://NVIDIAoLLAMA:11435"

# Limiti:
MAX_SAFE_TOKENS_NVIDIA = 4000
MAX_CONTEXT_TOKENS = 8192

# --- NUOVE FUNZIONI DI CONTROLLO PRE-ESECUZIONE ---

def check_pip_dependencies():
    """Verifica che le dipendenze Python necessarie siano installate."""
    print("🔎 Controllo delle dipendenze Python...")
    required_packages = ['ollama', 'pyyaml']
    
    for package in required_packages:
        try:
            # Tenta di importare il modulo
            __import__(package)
        except ImportError:
            print(f"❌ Errore: Il pacchetto '{package}' non è installato.")
            print("Per favore, installalo eseguendo: pip install ollama pyyaml")
            sys.exit(1)
            
    print("✅ Dipendenze Python verificate.")

def check_host_mapping():
    """Verifica che le mappature INTELoLLAMA e NVIDIAoLLAMA siano presenti in /etc/hosts."""
    print("🔎 Controllo della mappatura host (/etc/hosts)...")
    
    # Determina il percorso del file hosts
    if os.name == 'nt': # Windows
        hosts_path = r'C:\Windows\System32\drivers\etc\hosts'
    else: # Linux/macOS
        hosts_path = '/etc/hosts'
        
    required_hosts = ['INTELoLLAMA', 'NVIDIAoLLAMA']
    missing_hosts = []
    
    try:
        with open(hosts_path, 'r') as f:
            content = f.read()
            
            for host in required_hosts:
                # Verifica se la stringa '127.0.0.1 XXXoLLAMA' è presente nel file
                if f"127.0.0.1\t{host}" not in content and f"127.0.0.1 {host}" not in content:
                    missing_hosts.append(host)
            
            if missing_hosts:
                print(f"❌ Errore di Configurazione HOST: Le seguenti mappature mancano o non sono corrette in {hosts_path}:")
                for host in missing_hosts:
                    print(f"   - Manca: {host}")
                print("\nPer favore, aggiungi le seguenti righe (con privilegi di amministratore):")
                print("# Mappatura per i servizi Ollama")
                print("127.0.0.1    INTELoLLAMA")
                print("127.0.0.1    NVIDIAoLLAMA")
                sys.exit(1)
            
            print("✅ Mappatura host verificata.")
            
    except FileNotFoundError:
        print(f"❌ Errore: File hosts non trovato in {hosts_path}.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Errore durante la lettura del file hosts: {e}")
        sys.exit(1)


# --- 1. GENERAZIONE DEL FILE DOCKER-COMPOSE (RIMANE INVARIATA) ---

def generate_docker_compose():
    """Genera il contenuto del docker-compose.yml con le configurazioni GPU-only."""
    # ... (Il resto della funzione rimane invariato)
    
    compose_config = {
        'version': '3.8',
        'services': {
            'intel-llm': {
                'image': 'ollama/ollama',
                'container_name': 'ollama-intel',
                'ports': ['11434:11434'],
                'volumes': ['./ollama_data:/root/.ollama'],
                'environment': [
                    'OLLAMA_HOST=0.0.0.0',
                    'OLLAMA_GPU_LAYERS=0',
                    f'OLLAMA_MODEL={MODEL_SLM_INTEL}'
                ],
            },
            'nvidia-calc': {
                'image': 'ollama/ollama',
                'container_name': 'ollama-nvidia',
                'ports': ['11435:11434'],
                'volumes': ['./ollama_data:/root/.ollama'],
                'environment': [
                    'OLLAMA_HOST=0.0.0.0',
                    'OLLAMA_GPU_LAYERS=100',
                    'OLLAMA_KV_CACHE_TYPE=Q8',
                    'OLLAMA_FLASH_ATTENTION=true',
                    f'OLLAMA_MODEL={MODEL_SLM_NVIDIA}'
                ],
                'deploy': {
                    'resources': {
                        'reservations': {
                            'devices': [
                                {
                                    'driver': 'nvidia',
                                    'count': '1',
                                    'capabilities': ['gpu']
                                }
                            ]
                        }
                    }
                }
            }
        }
    }
    
    try:
        with open('docker-compose.yml', 'w') as f:
            yaml.dump(compose_config, f, default_flow_style=False)
        print("✅ File 'docker-compose.yml' generato con successo.")
    except Exception as e:
        print(f"❌ Errore nella scrittura di docker-compose.yml: {e}")
        sys.exit(1)


# --- 2. GESTIONE E AVVIO DOCKER (RIMANE INVARIATA) ---

def manage_docker_compose(action: str):
    # ... (Il corpo della funzione rimane invariato)
    try:
        if action == 'up':
            print("🚀 Avvio dei container Ollama (INTELoLLAMA e NVIDIAoLLAMA)...")
            subprocess.run(['docker', 'compose', 'up', '-d'], check=True)
            print("⏳ Attesa per l'avvio dei servizi Ollama e il download dei modelli...")
            time.sleep(30)
            
            print(f"Scarico i modelli: {MODEL_SLM_INTEL} e {MODEL_SLM_NVIDIA}...")
            subprocess.run(['ollama', 'pull', MODEL_SLM_INTEL], check=False)
            subprocess.run(['ollama', 'pull', MODEL_SLM_NVIDIA], check=False)
            
        elif action == 'down':
            print("🛑 Spegnimento e pulizia dei container...")
            subprocess.run(['docker', 'compose', 'down'], check=True)
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Errore durante l'esecuzione del comando Docker: {e}")
        sys.exit(1)


# --- 3. LOGICA DELL'ORCHESTRATORE (TASK MANAGEMENT) (RIMANE INVARIATA) ---

def estimate_tokens(prompt: str) -> int:
    # ... (Il corpo della funzione rimane invariato)
    return int(len(prompt.split()) * 1.3)

def run_llm_task(prompt: str, client_nvidia: ollama.Client, client_intel: ollama.Client):
    # ... (Il corpo della funzione rimane invariato)
    prompt_tokens = estimate_tokens(prompt)
    print(f"\n--- Analisi Prompt: {prompt_tokens} token ---")

    # A. Fall-back iniziale: Se il prompt è troppo lungo per la VRAM sicura (4GB)
    if prompt_tokens > MAX_SAFE_TOKENS_NVIDIA:
        print(f"⚠️ Prompt troppo lungo. Uso SLM (INTELoLLAMA) come fallback iniziale.")
        try:
            response = client_intel.generate(
                model=MODEL_SLM_INTEL, 
                prompt=prompt,
                options={"num_ctx": MAX_CONTEXT_TOKENS}
            )
            return f"[RISPOSTA SLM/INTEL] {response['response']}"
        except Exception as e:
             return f"Errore critico SLM (INTELoLLAMA) - Impossibile eseguire: {e}"


    # B. Esecuzione Primaria: GPU NVIDIA (Calcolo Veloce)
    print(f"✅ Tentativo su GPU NVIDIA ({prompt_tokens} token).")
    try:
        response = client_nvidia.generate(
            model=MODEL_SLM_NVIDIA, 
            prompt=prompt,
            options={
                "num_ctx": MAX_CONTEXT_TOKENS,
                "temperature": 0.7 
            }
        )
        return f"[RISPOSTA SLM/NVIDIA] {response['response']}"
        
    except OllamaAPIError as e:
        error_msg = str(e).lower()
        if "out of memory" in error_msg or "context exceeded" in error_msg:
            print(f"❌ Errore VRAM su NVIDIA (cache piena). Fall-back d'emergenza su SLM (INTELoLLAMA).")
            try:
                response = client_intel.generate(
                    model=MODEL_SLM_INTEL, 
                    prompt=prompt,
                    options={"num_ctx": MAX_CONTEXT_TOKENS}
                )
                return f"[RISPOSTA DI EMERGENZA SLM] {response['response']}"
            except Exception as intel_e:
                return f"Errore critico SLM (INTELoLLAMA) dopo fall-back: {intel_e}"
        
        raise e 

def main_orchestration_loop(client_nvidia: ollama.Client, client_intel: ollama.Client):
    # ... (Il corpo della funzione rimane invariato)
    print("\n--- INIZIO CICLO DI ORCHESTRAZIONE DEL CARICO ---")

    # CARICO 1: Compito Semplice (Breve)
    task1_prompt = "Spiega, in termini brevi, il concetto di ridondanza nella teoria dell'informazione di Shannon."
    print(f"\n[CARICO 1] - Compito breve (dovrebbe usare NVIDIA).")
    result1 = run_llm_task(task1_prompt, client_nvidia, client_intel)
    print("\n[RISULTATO 1]\n", result1)

    # CARICO 2: Compito Complesso/Lungo (Forza Fall-back SLM/Intel)
    task2_prompt = "Scrivi una breve sinossi su come la limitazione della memoria VRAM influenzi il design di un'architettura a microservizi basata su Large Language Models." * 4
    print(f"\n[CARICO 2] - Compito estremamente lungo (Forza SLM/INTELoLLAMA).")
    result2 = run_llm_task(task2_prompt, client_nvidia, client_intel)
    print("\n[RISULTATO 2]\n", result2)

# --- FUNZIONE PRINCIPALE ---

if __name__ == "__main__":
    
    # Esegue i controlli prima di procedere con Docker
    check_pip_dependencies()
    check_host_mapping()
    
    # 0. Pulizia Precedente e Generazione File
    manage_docker_compose('down')
    generate_docker_compose()
    
    # 1. Avvio dei Container
    manage_docker_compose('up')
    
    # 2. Inizializzazione Client API (Usando i nomi host personalizzati)
    client_intel = ollama.Client(host=HOST_INTEL)
    client_nvidia = ollama.Client(host=HOST_NVIDIA)
    
    # 3. Esecuzione del loop di orchestrazione
    main_orchestration_loop(client_nvidia, client_intel)

    # 4. Spegnimento Container
    print("\n--- FINE ORCHESTRAZIONE ---")
    manage_docker_compose('down')
