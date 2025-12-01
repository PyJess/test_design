from typing import Optional, List, Union
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv
import requests
import json

load_dotenv()


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

CLIENT_ID = os.getenv('CLIENT_ID')
CLIENT_SECRET = os.getenv('CLIENT_SECRET')


def get_token():

    url = os.getenv('BASE_URL_TOKEN')
    payload = {
        'client_id': CLIENT_ID,
        'client_secret': CLIENT_SECRET,
        'scope': 'api://3bb3eccb-0787-4526-811e-ec3dab677121/.default',
        'grant_type': 'client_credentials'
    }
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}

    response = requests.post(url, data=payload, headers=headers)
    if response.status_code == 200:
        return response.json().get('access_token')
    else:
        print(f"Error Token: {response.status_code} - {response.text}")
        return None


class LLMClient:
    def __init__(
            self,
            model_name: str = "gpt-4o-mini",
            embedder_model_name: str = "text-embedding-3-large",
            temperature=0,
            # max_tokens=1000,
            # top_p=1
    ):
        self.model_name = model_name
        token = self._authenticate()
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Ocp-Apim-Subscription-Key": os.getenv("OCP_APIM_SUBSCRIPTION_KEY")
        }
        self.openai_api_base = os.getenv("OPENAI_BASE_URL")
        self.temperature = temperature
        self.embedding_deployment = embedder_model_name
        self.PROJECT_ID = os.getenv("PROJECT_ID", "default")
        try:
            self._client = ChatOpenAI(
                model=self.model_name,
                default_headers=self.headers,
                openai_api_base=self.openai_api_base + "/aw",
                temperature=self.temperature,
                api_key="useless",
            )

            self.embedder = OpenAIEmbeddings(
                model=self.embedding_deployment,  # just for metadata
                base_url=f"{self.openai_api_base}/deployments/{self.embedding_deployment}",
                default_headers=self.headers,
                api_key="useless",
            )
        except Exception as e:
            print(f"[LLMClient] initialization error: {e}")
            self._client = None

    def __getattr__(self, item):
        return getattr(self._client, item)

    def _authenticate(self) -> str:
        """Retrieve token."""
        return get_token()

    def available(self) -> bool:
        return self._client is not None

    def get_embedding(self, text:str)->Union[List[float], None]:

        """
        Returns the embedding vector for the given text.
        """

        url = f"{self.openai_api_base}/deployments/{self.embedding_deployment}/embeddings?project={self.PROJECT_ID}"

        payload = {
            "user_input": [text]
        }

        try:
            # Send request
            response = requests.post(url, headers=self.headers, json=payload)
            # Since we pass single query, we get index '0'
            return response.json()["output"][0]

        except Exception as e:
            print(f"[LLMClient] error getting embedding: {e}")
            return None
        
        
        


# Nel file pdf_data_extraction.py, dopo agent_data_extraction
llm_client = LLMClient()
def add_embeddings_to_json(parsed_data: dict, output_json_path: str, embed_field: str = "titolo_e_contenuto"):
    """
    Aggiunge embeddings al JSON esistente o ai dati parsati, 
    usando il client LLM già inizializzato (llm_client).
    """
    global llm_client # Usa l'istanza globale
    
    sezioni = parsed_data.get("sezioni", [])
    if not sezioni:
        print("Nessuna sezione trovata per l'embedding.")
        return parsed_data
        
    print(f"[INFO] Trovate {len(sezioni)} sezioni da processare per l'embedding.")
    
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
        
        # Genera embedding usando il metodo del tuo LLMClient
        # Nota: La funzione get_embedding in embedding.py NON è asincrona.
        embedding = llm_client.get_embedding(text_to_embed) 
        
        if embedding:
            # Aggiungi l'embedding alla sezione
            sezione["embedding"] = embedding
            # sezione["embedding_model"] = llm_client.embedding_deployment if hasattr(llm_client, 'embedding_deployment') else "Unknown"
            sezione["embedding_field"] = embed_field
        else:
            print(f"⚠️ Impossibile generare embedding per la sezione {i}.")

    # Salvataggio JSON finale
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(parsed_data, f, ensure_ascii=False, indent=2)
    
    print(f"JSON con embeddings salvato in: {output_json_path}")
    
    return parsed_data

# from pathlib import Path
# input_test = r"C:\Users\BencichWilliam\Desktop\test_design_xhesina\test_design\pdf_reader\regolamento_strutturato.json"


# BASE_DIR = Path(r"C:\Users\BencichWilliam\Desktop\test_design_xhesina\test_design\pdf_reader")

# OUTPUT_FILENAME = "embedding_regolamento_strutturato.json" 

# output_json_path = BASE_DIR / OUTPUT_FILENAME


    
# parsed_data = load_json(input_test)
            
#             # 3. Esegui la funzione con gli argomenti corretti
#             # Nota: la funzione utilizza l'istanza globale 'llm_client'
# embedded_data = add_embeddings_to_json(
# parsed_data=parsed_data,
# output_json_path=str(output_json_path),
# embed_field="titolo_e_contenuto" 
# )

# print(f"Esempio embedding (primi 5 valori): {embedded_data['sezioni'][0]['embedding'][:5]}")
