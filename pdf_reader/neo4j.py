from neo4j import GraphDatabase
import os

# Variabili d'ambiente per Neo4j (da aggiungere al tuo .env se non le hai)
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "la_tua_password")

# --- Nuovo Client Neo4j ---
class Neo4jClient:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def execute_query(self, query, parameters=None):
        with self.driver.session(database="document-file") as session:
            result = session.run(query, parameters)
            return [record for record in result]
# -------------------------

# Inizializza il client Neo4j (dopo llm_client)
neo4j_client = Neo4jClient(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)


# Query Cypher per creare l'indice vettoriale
CREATE_INDEX_QUERY = """
CREATE VECTOR INDEX `document-index` IF NOT EXISTS
FOR (s:Section) ON s.embedding
OPTIONS {
  nodeProperty: 'embedding', 
  indexContainer: true, 
  vectorDimensions: 1536, 
  similarityFunction: 'cosine'
}
"""
neo4j_client.execute_query(CREATE_INDEX_QUERY)
print("[INFO] Indice vettoriale 'document-index' creato/verificato.")


def load_embeddings_to_neo4j(parsed_data: dict):
    # Rimuovi l'uso di global llm_client se lo fai in un'altra funzione
    
    sezioni = parsed_data.get("sezioni", [])
    if not sezioni:
        print("Nessuna sezione trovata per l'embedding.")
        return
        
    print(f"[INFO] Trovate {len(sezioni)} sezioni da processare per il caricamento in Neo4j.")
    
    # Query Cypher per la creazione del nodo
    CREATE_NODE_QUERY = """
    UNWIND $sections as section_data
    CREATE (s:Section {
        titolo: section_data.titolo,
        contenuto: section_data.contenuto,
        embedding_field: section_data.embedding_field,
        # L'embedding deve essere passato come List[float]
        embedding: section_data.embedding
    })
    RETURN count(s) AS nodes_created
    """
    
    # 1. Calcola l'embedding (come già fai)
    # 2. Carica i dati nella lista
    sections_for_neo4j = []
    
    # DEVI USARE LA TUA LOGICA AGGIORNATA DI EMBEDDING QUI!
    # userai la logica di add_embeddings_to_json per popolare `sections_for_neo4j`
    # Esempio:
    for i, sezione in enumerate(sezioni):
        # ... (Logica per ottenere text_to_embed)
        # ... (Logica per chiamare embedding = llm_client.get_embedding(text_to_embed))
        
        if embedding:
            sezione["embedding"] = embedding # Aggiungi al dizionario
            sections_for_neo4j.append(sezione)
        
    
    # 3. Esegui il caricamento batch in Neo4j
    result = neo4j_client.execute_query(
        CREATE_NODE_QUERY, 
        parameters={"sections": sections_for_neo4j}
    )
    
    print(f"[✅] Caricamento completato. Nodi creati: {result[0]['nodes_created'] if result else 0}")
    
    
 def semantic_search_neo4j(query: str, top_k: int = 5, similarity_threshold: float = 0.0):
    """
    Esegue la ricerca semantica usando l'indice vettoriale di Neo4j.
    """
    print(f"[INFO] Generazione embedding per la query: '{query}'")
    # Usa la TUA funzione di embedding esistente (da embedding.py o pdf_extraction.py)
    query_embedding = llm_client.get_embedding(query) 
    
    if query_embedding is None:
        print("❌ Impossibile generare embedding per la query.")
        return []

    # Query Cypher: usa db.index.vector.queryNodes per la ricerca vettoriale
    SEARCH_QUERY = """
    CALL db.index.vector.queryNodes(
        'document-index', 
        $top_k, 
        $query_vector
    ) YIELD node, score
    WHERE score >= $threshold
    RETURN node.titolo AS titolo, 
           node.contenuto AS contenuto, 
           score AS similarity_score
    ORDER BY score DESC
    """
    
    results = neo4j_client.execute_query(
        SEARCH_QUERY,
        parameters={
            "top_k": top_k,
            "query_vector": query_embedding,
            "threshold": similarity_threshold
        }
    )
    
    # Mappa i risultati per la visualizzazione
    return [
        {
            "titolo": r["titolo"],
            "contenuto": r["contenuto"],
            "similarity_score": r["similarity_score"]
        }
        for r in results
    ]

# --- ESEMPIO DI TEST ---
if __name__ == "__main__":
    # ... (assicurati che Neo4jClient e llm_client siano inizializzati)
    
    # 1. (Optional) Ricarica i dati con embedding per assicurarsi che siano in Neo4j
    # parsed_data = load_json(input_test)
    # load_embeddings_to_neo4j(parsed_data) 
    
    # 2. Esegui la ricerca
    query = "Regole riguardanti la giocata"
    
    results = semantic_search_neo4j(query=query, top_k=3)
    
    display_results(results, show_content_chars=200)