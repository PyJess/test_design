import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Any, Callable, Union

# Importa LLMClient dal tuo modulo per ottenere l'embedding della query
from embedding import LLMClient
# Inizializza il client LLM per l'embedding della query (se non lo fai globalmente altrove)
# *Assicurati che LLMClient sia in grado di generare embeddings*
llm_client = LLMClient() 

def get_query_embedding(text: str) -> Union[List[float], None]:
    """
    Wrapper che usa il metodo get_embedding dell'istanza LLMClient
    per generare l'embedding della query.
    """
    return llm_client.get_embedding(text)

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calcola la similarità del coseno tra due vettori.
    """
    vec1_np = np.array(vec1)
    vec2_np = np.array(vec2)

    dot_product = np.dot(vec1_np, vec2_np)
    norm_vec1 = np.linalg.norm(vec1_np)
    norm_vec2 = np.linalg.norm(vec2_np)

    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0

    return dot_product / (norm_vec1 * norm_vec2)


# --- FUNZIONE DI RICERCA VETTORIALE ---
def semantic_search(
    query: str, 
    json_path: Path, 
    top_k: int = 5, 
    similarity_threshold: float = 0.0
) -> List[Dict[str, Any]]:
    """
    Esegue una ricerca semantica sul JSON con embeddings utilizzando la similarità del coseno.
    
    Args:
        query: La domanda/ricerca dell'utente.
        json_path: Path al JSON con embeddings salvati.
        get_embed_func: La funzione (dal LLMClient) per generare l'embedding della query.
        top_k: Numero di risultati da restituire.
        similarity_threshold: Soglia minima di similarità (0-1).
    
    Returns:
        Lista di dizionari con sezioni ordinate per similarità.
    """
    print(f"[INFO] Caricamento JSON: {json_path}")
    
    # 1. Carica il JSON con embeddings
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f" Errore nel caricamento del JSON: {e}")
        return []
    
    # Il tuo JSON strutturato ha la chiave "sezioni"
    sezioni = data.get("sezioni", []) 
    
    if not sezioni or "embedding" not in sezioni[0]:
        print("Nessuna sezione o embedding trovata nel JSON. Assicurati che il file sia corretto.")
        return []
    
    # 2. Genera embedding per la query
    print(f"[INFO] Generazione embedding per la query: '{query}'")
    query_embedding = get_query_embedding(query)
    
    if query_embedding is None:
        print("Impossibile generare embedding per la query. Verifica la connessione API.")
        return []
    
    print(f"[INFO] Calcolo similarità (Cosine Similarity) per {len(sezioni)} sezioni...")
    
    # 3. Calcola similarità per ogni sezione
    results = []
    for i, sezione in enumerate(sezioni):
        section_embedding = sezione.get("embedding")
        
        if section_embedding is None:
            continue
        
        similarity = cosine_similarity(query_embedding, section_embedding)
        
        # 4. Filtra per soglia e registra
        if similarity >= similarity_threshold:
            results.append({
                "titolo": sezione.get("titolo", "Nessun titolo"),
                "contenuto": sezione.get("contenuto", "Nessun contenuto"),
                "similarity_score": float(similarity),
                "index": i
            })
    
    # 5. Ordina e restituisci top K
    results.sort(key=lambda x: x["similarity_score"], reverse=True)
    
    return results[:top_k]


def display_results(results: List[Dict[str, Any]], show_content_chars: int = 300):
    """Visualizza i risultati della ricerca in modo leggibile."""
    
    if not results:
        print("\nNessun risultato trovato")
        return
    
    print(f"\n{'='*80}")
    print(f"RISULTATI RICERCA - {len(results)} sezioni trovate")
    print(f"{'='*80}\n")
    
    for i, result in enumerate(results, start=1):
        score = result["similarity_score"]
        titolo = result["titolo"]
        contenuto = result["contenuto"]
        
      
        
        print(f"{i}.  {titolo}")
        print(f"   Similarità: {score:.4f} {score*100:.1f}%")
        print(f"   {'-'*76}")
        
        content_preview = contenuto[:show_content_chars]
        if len(contenuto) > show_content_chars:
            content_preview += "..."
        
        print(f"   {content_preview}")
        print(f"   {'='*76}\n")
        
        if not contenuto:
                print("CONTENUTO VUOTO: Il documento JSON ha un campo 'contenuto' vuoto per questa sezione.")
        else:
                # Mostra anteprima contenuto
                content_preview = contenuto.strip()[:show_content_chars]
                if len(contenuto.strip()) > show_content_chars:
                    content_preview += "..."
                
                # Stampa il contenuto troncato in una riga chiara
                print(f"   Preview: {content_preview.replace('\\n', ' ')}") 
            # --- FINE BLOCCO DI STAMPA CONTENUTO ---
            
                print(f"   {'='*76}\n")
# if __name__ == "__main__":
#     # --- Configurazione e Test ---
    
#     BASE_DIR = Path(__file__).parent
#     JSON_FILENAME = "embedding_regolamento_strutturato.json" #
#     json_path = BASE_DIR / JSON_FILENAME
    
#     if not json_path.exists():
#         print(f" ERRORE: File con embeddings non trovato in → {json_path}")
#         print("Assicurati di aver generato il file JSON con gli embeddings.")
#     else:
#         queries = ["Posta di gioco"]

#         for query in queries:
#             print(f"\n{'#'*80}")
#             print(f"RICERCA VETTORIALE: {query}")
#             print(f"{'#'*80}")
            
#             # Esegui la ricerca, passando il metodo get_embedding del tuo LLMClient
#             results = semantic_search(
#                 query=query,
#                 json_path=json_path,
#                 top_k=5, 
#                 similarity_threshold=0.3  # Soglia minima di similarità
#             )
            
#             display_results(results, show_content_chars=200)