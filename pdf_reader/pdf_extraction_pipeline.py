import asyncio
import json
from pathlib import Path
from typing import List, Dict, Any

from pdf_data_extraction import agent_data_extraction 
from embedding import add_embeddings_to_json, LLMClient  
from pdf_vector_search import semantic_search, display_results
from dotenv import load_dotenv

load_dotenv() 

llm_client =LLMClient()
# --- CONFIGURAZIONE PATH ---
# Definisci la directory base dove verranno salvati i file JSON
BASE_DIR = Path(__file__).parent 
OUTPUT_JSON_FILENAME = "regolamento_strutturato.json"
EMBEDDED_JSON_FILENAME = "embedding_regolamento_strutturato.json"


async def run_extraction_and_search_pipeline(
    pdf_path: Path, 
    user_query: str, 
    force_reprocess: bool = False,
    top_k: int = 5,
    similarity_threshold: float = 0.3
) -> List[Dict[str, Any]]:
    """
    Pipeline completa: Estrazione PDF, Embedding del Documento, e Vector Search della Query.

    Args:
        pdf_path: Path del file PDF da analizzare.
        user_query: La query di ricerca dell'utente.
        force_reprocess: Se True, forza la ri-estrazione e ri-embedding del PDF.
        top_k: Numero massimo di risultati da restituire nella Vector Search.
        similarity_threshold: Soglia minima di similarità del coseno.
        
    Returns:
        Lista dei risultati della Vector Search (sezioni contestuali).
    """
    
    output_json_path = BASE_DIR / OUTPUT_JSON_FILENAME
    embedded_json_path = BASE_DIR / EMBEDDED_JSON_FILENAME

    print(f"--- 1. INIZIO PIPELINE ---")
    print(f"PDF: {pdf_path.name} | Query: '{user_query}'")
    print("-------------------------")

    ## 1. Agente di Estrazione (PDF -> JSON Strutturato)
    if not output_json_path.exists() or force_reprocess:
        print(f"## 1. AGENTE: Estrazione e Strutturazione JSON (richiesto/forzato) ##")
        try:
            # agent_data_extraction è asincrona
            await agent_data_extraction(str(pdf_path), str(output_json_path))
            print("JSON Strutturato creato con successo.")
        except Exception as e:
            print(f"Errore nell'estrazione del PDF: {e}")
            return []
    else:
        print(f"Salto Fase 1: JSON Strutturato già esistente in {output_json_path.name}")
        
        # Carica il JSON esistente per la fase successiva
        try:
            with open(output_json_path, 'r', encoding='utf-8') as f:
                parsed_data = json.load(f)
        except Exception as e:
            print(f"Errore nel caricamento del JSON: {e}")
            return []
            
            
    ## 2. Agente di Embedding (Aggiunta vettori al JSON)
    # Verifichiamo se il file embeddato esiste o se l'estrazione è stata forzata
    if not embedded_json_path.exists() or force_reprocess or not parsed_data:
        print("\n## 2. AGENTE: Embedding del Documento ##")
        
        # Se non forzato, carica il JSON strutturato dalla Fase 1
        if not parsed_data:
            parsed_data = load_json(output_json_path)
            if not parsed_data: return [] # Esci se non si riesce a caricare

        add_embeddings_to_json(
            parsed_data=parsed_data,
            output_json_path=str(embedded_json_path),
            embed_field="titolo_e_contenuto" # Usa il campo di embedding scelto
        )
        print("Documento embeddato e salvato con successo.")
    else:
        print(f"Salto Fase 2: JSON con Embeddings già esistente in {embedded_json_path.name}")


    ## 3. Funzione Vector Search (Embed query e cerca)
    print("\n## 3. FUNZIONE: Vector Search e Retrieval ##")
    
    # semantic_search è sincrona e legge direttamente il file embeddato
    results = semantic_search(
        query=user_query,
        json_path=embedded_json_path,
        top_k=top_k,
        similarity_threshold=similarity_threshold
    )
    
    if results:
        display_results(results, show_content_chars=200)
    else:
        print("Nessun risultato di ricerca trovato con i parametri specificati.")
        
    return results


def load_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"    Errore nel caricamento del JSON: {e}")
        return None


# --- ESECUZIONE DI TEST (Come richiesto) ---
# if __name__ == "__main__":
    
#     PDF_INPUT_PATH = BASE_DIR / "20210930_REGOLAMENTO_SVT_2021.pdf" 
#     pdf_test = BASE_DIR / "20210930_REGOLAMENTO_SVT_2021.pdf" # Esempio
#     if not PDF_INPUT_PATH.exists():
#         print(f"\nERRORE: File PDF non trovato: {PDF_INPUT_PATH}")
#         print("Modifica PDF_INPUT_PATH con il percorso corretto del tuo file PDF.")
#     else:
#         # Queries di esempio (potranno essere prese dall'excel nel Main)
#         test_queries = [
#             " posta di gioco.",
#             "Termini condizione",
#         ]
        
#         # Esegui la pipeline per tutte le query
#         for i, query in enumerate(test_queries, 1):
#             print(f"\n\n{'='*100}")
#             print(f"CICLO DI TEST {i}: Esecuzione RAG per la query: {query}")
#             print(f"{'='*100}")
            
#             # Esegui la pipeline (uso asyncio.run poiché agent_data_extraction è asincrona)
#             asyncio.run(run_extraction_and_search_pipeline(
#                 pdf_path=pdf_test,
#                 user_query=query,
#                 force_reprocess=False, # Imposta a True se devi rifare l'estrazione e embedding
#                 top_k=1
#             ))