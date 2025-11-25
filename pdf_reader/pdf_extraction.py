import os
import json
import fitz  
from docx import Document
from dotenv import load_dotenv
from pathlib import Path
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate
from openai import OpenAI
import numpy as np

load_dotenv()
client = OpenAI()


def read_document(file_path):
    # Convert Path object to string if necessary
    file_path = str(file_path)
    
    if file_path.lower().endswith(".pdf"):
        text = ""
        with fitz.open(file_path) as doc:
            for page in doc:
                text += page.get_text("text")
        return text
    # elif file_path.lower().endswith(".docx"):
    #     doc = Document(file_path)
    #     return "\n".join([p.text.strip() for p in doc.paragraphs if p.text.strip()])
    else:
        raise ValueError("Formato non supportato. Usa .pdf o .docx")

response_schemas = [
    ResponseSchema(
        name="sezioni",
        description=(
            "Lista di oggetti, ciascuno con 'titolo' e 'contenuto'. "
            "Ogni titolo può essere un articolo, paragrafo, capitolo, sezione o altro livello identificabile nel testo. "
            "Il contenuto è il testo completo associato a quel titolo."
        ),
    )
]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()



def extract_structure(file_path, output_json_path):
    print(f"[INFO] Lettura file: {file_path}")
    text = read_document(file_path)
    text = text[:18000]  # Limite sicurezza

    # Prompt per GPT
    prompt = PromptTemplate(
        template=(
            "Analizza il seguente documento e suddividilo in sezioni logiche.\n"
            "Identifica automaticamente tutti i titoli (es. 'Articolo 1', 'Paragrafo 2.1', 'Sezione III', 'Titolo I', 'Capitolo 5', ecc.) "
            "e associa a ciascuno il testo che gli appartiene.\n\n"
            "Restituisci un JSON di questa forma:\n"
            "[{{ 'titolo': 'Titolo o intestazione della sezione', 'contenuto': 'Testo completo associato' }}, ...]\n\n"
            "{format_instructions}\n\n"
            "Documento:\n{text}"
        ),
        input_variables=["text"],
        partial_variables={"format_instructions": format_instructions},
    )

    formatted_prompt = prompt.format(text=text)

    print("[INFO] Analisi con LLM in corso...")

    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": "Sei un assistente esperto nella strutturazione di documenti in sezioni."},
            {"role": "user", "content": formatted_prompt},
        ],
        response_format={"type": "json_object"},
    )

    # Parsing dell’output
    try:
        parsed = output_parser.parse(response.choices[0].message.content)
    except Exception:
        parsed = json.loads(response.choices[0].message.content)

    # Salvataggio JSON
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(parsed, f, ensure_ascii=False, indent=2)

    print(f"[✅] File JSON salvato in: {output_json_path}")
    return parsed


input_path = Path(__file__).parent/"20210930_REGOLAMENTO_SVT_2021.pdf"
output_path = Path(__file__).parent.parent/"pdf_reader"/"regolamento_structured.json"  


# result = extract_structure(input_path, output_path)



def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_embedding(text, model="text-embedding-3-small"):
    """Ottiene l'embedding di un testo usando OpenAI API"""
    text = text.replace("\n", " ")  # Rimuovi newline
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding


def add_embeddings_to_json(input_json_path, output_json_path, embed_field="titolo_e_contenuto"):
    """
    Aggiunge embeddings al JSON esistente
    
    Args:
        input_json_path: Path del JSON originale
        output_json_path: Path dove salvare il JSON con embeddings
        embed_field: Campo da embeddare - opzioni:
            - "titolo": solo titolo
            - "contenuto": solo contenuto
            - "titolo_e_contenuto": concatena titolo + contenuto (default)
    """
    print(f"[INFO] Caricamento JSON da: {input_json_path}")
    
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    sezioni = data.get("sezioni", [])
    print(f"[INFO] Trovate {len(sezioni)} sezioni da processare")
    
    # Aggiungi embeddings a ciascuna sezione
    for i, sezione in enumerate(sezioni, start=1):
        titolo = sezione.get("titolo", "")
        contenuto = sezione.get("contenuto", "")
        
        # Determina il testo da embeddare
        if embed_field == "titolo":
            text_to_embed = titolo
        elif embed_field == "contenuto":
            text_to_embed = contenuto
        else:  # "titolo_e_contenuto"
            text_to_embed = f"{titolo}\n\n{contenuto}"
        
        print(f"[INFO] Generazione embedding per sezione {i}/{len(sezioni)}: {titolo[:50]}...")
        
        # Genera embedding
        embedding = get_embedding(text_to_embed)
        
        # Aggiungi l'embedding alla sezione
        sezione["embedding"] = embedding
        sezione["embedding_model"] = "text-embedding-3-small"
        sezione["embedding_field"] = embed_field
    
    # Salva il nuovo JSON
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"[✅] JSON con embeddings salvato in: {output_json_path}")
    print(f"[INFO] Dimensione embedding: {len(sezioni[0]['embedding'])} dimensioni")
    
    return data


# Esempio di utilizzo
input_path = Path(__file__).parent /"regolamento_structured.json"
output_path = Path(__file__).parent /"embedded.json"

if not input_path.exists():
    print(f"❌ ERRORE: File non trovato → {input_path}")
else:
    print("--")
    # Opzioni per embed_field:
    # - "titolo": embedda solo il titolo
    # - "contenuto": embedda solo il contenuto
    # - "titolo_e_contenuto": embedda titolo + contenuto insieme
    # result = add_embeddings_to_json(
    #     input_path, 
    #     output_path,
    #     embed_field="titolo_e_contenuto"  # Modifica secondo necessità
    # )
    
    # Verifica il risultato
#     print("\n[INFO] Esempio di sezione con embedding:")
#     sezione_esempio = result["sezioni"][0]
#     print(f"Titolo: {sezione_esempio['titolo']}")
#     print(f"Contenuto (primi 100 char): {sezione_esempio['contenuto'][:100]}...")
#     print(f"Embedding (prime 5 dimensioni): {sezione_esempio['embedding'][:5]}")
# file_json = load_json(Path(__file__).parent/"regolamento_structured.json")

# Direct access - sezioni is already a list


def cosine_similarity(vec1, vec2):
    """
    Calcola la similarità del coseno tra due vettori

    Formula: cos(θ) = (A · B) / (||A|| * ||B||)
    Risultato: valore tra -1 e 1 (1 = identici, 0 = ortogonali, -1 = opposti)
    """
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)

    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0

    return dot_product / (norm_vec1 * norm_vec2)

def semantic_search(query, json_path, top_k=5, similarity_threshold=0.0):
    """
    Esegue una ricerca semantica sul JSON con embeddings
    
    Args:
        query: La domanda/ricerca dell'utente
        json_path: Path al JSON con embeddings
        top_k: Numero di risultati da restituire
        similarity_threshold: Soglia minima di similarità (0-1)
    
    Returns:
        Lista di dizionari con sezioni ordinate per similarità
    """
    print(f"[INFO] Caricamento JSON: {json_path}")
    
    # Carica il JSON con embeddings
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    sezioni = data.get("sezioni", [])
    
    if not sezioni:
        print("❌ Nessuna sezione trovata nel JSON")
        return []
    
    # Verifica che ci siano embeddings
    if "embedding" not in sezioni[0]:
        print("❌ Il JSON non contiene embeddings. Esegui prima add_embeddings_to_json()")
        return []
    
    print(f"[INFO] Generazione embedding per la query: '{query}'")
    query_embedding = get_embedding(query)
    
    print(f"[INFO] Calcolo similarità per {len(sezioni)} sezioni...")
    
    # Calcola similarità per ogni sezione
    results = []
    for i, sezione in enumerate(sezioni):
        section_embedding = sezione.get("embedding")
        
        if section_embedding is None:
            continue
        
        # Calcola similarità del coseno
        similarity = cosine_similarity(query_embedding, section_embedding)
        
        # Filtra per soglia
        if similarity >= similarity_threshold:
            results.append({
                "titolo": sezione.get("titolo", ""),
                "contenuto": sezione.get("contenuto", ""),
                "similarity_score": float(similarity),
                "index": i
            })
    
    # Ordina per similarità decrescente
    results.sort(key=lambda x: x["similarity_score"], reverse=True)
    
    # Restituisci top K risultati
    return results[:top_k]

def display_results(results, show_content_chars=300):
    """Visualizza i risultati della ricerca in modo leggibile"""
    if not results:
        print("\n⚠️ Nessun risultato trovato")
        return
    
    print(f"\n{'='*80}")
    print(f"🔍 RISULTATI RICERCA - {len(results)} sezioni trovate")
    print(f"{'='*80}\n")
    
    for i, result in enumerate(results, start=1):
        score = result["similarity_score"]
        titolo = result["titolo"]
        contenuto = result["contenuto"]
        
        # Barra di progresso visuale per il punteggio
        bar_length = 20
        filled = int(bar_length * score)
        bar = "█" * filled + "░" * (bar_length - filled)
        
        print(f"{i}. 📄 {titolo}")
        print(f"   Similarità: {score:.4f} [{bar}] {score*100:.1f}%")
        print(f"   {'-'*76}")
        
        # Mostra anteprima contenuto
        content_preview = contenuto[:show_content_chars]
        if len(contenuto) > show_content_chars:
            content_preview += "..."
        
        print(f"   {content_preview}")
        print(f"   {'='*76}\n")
        
    
    
if __name__ == "__main__":
    # Path al JSON con embeddings
    json_path = Path(__file__).parent / "embedded.json"
    
    if not json_path.exists():
        print(f"❌ File non trovato: {json_path}")
        print("Esegui prima lo script per generare gli embeddings!")
    else:
        # Query di esempio
        queries = [
          "Posta di gioco, combinazione di gioco"
        ]
        
        # Prova diverse query
        for query in queries:
            print(f"\n{'#'*80}")
            print(f"QUERY: {query}")
            print(f"{'#'*80}")
            
            results = semantic_search(
                query=query,
                json_path=json_path,
                top_k=3,  # Top 3 risultati
                similarity_threshold=0.3  # Minimo 30% di similarità
            )
            
            display_results(results, show_content_chars=200)
            
            # Pausa tra le query (opzionale)
            input("Premi INVIO per continuare con la prossima query...\n")
