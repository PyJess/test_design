import base64
from fastapi import FastAPI, HTTPException, Form, WebSocket, UploadFile, File
import math
import sys
import os
from fastapi import Body
from pydantic import BaseModel
import shutil
# from langchain_openai import OpenAIEmbeddings
# from langchain.vectorstores import FAISS
from pathlib import Path
from langgraph.graph import StateGraph
from utils.state import TestDesignState
from utils.cryptographer import decrypt_payload, encrypt_payload
from utils.logging_ws import active_connections
import asyncio
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Input_extraction.extract_polarion_field_mapping import *
from utils.simple_functions import *
# from llm.llm import a_invoke_model
from Processing.test_design import *
from llm.llm import LLMClient
from fastapi.middleware.cors import CORSMiddleware


llm_client = LLMClient()

app = FastAPI(title="Controllo Sintattico Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000","https://ea-seneca-dev-app-service.sisal.it","https://ea-seneca-dev-app-service.azurewebsites.net"],  # React in dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/ws/logs")
async def websocket_logs(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except Exception:
        pass
    finally:
        if websocket in active_connections:
            active_connections.remove(websocket)

def make_serializable(obj, top_level=True):
    """Convert objects to JSON-serializable format"""

    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')
    

    if isinstance(obj, pd.Series):
        return obj.to_dict()
    

    if isinstance(obj, dict):
        if top_level:

            obj = {k: v for k, v in obj.items() if k in [
                "test_cases", "ai_results", "merged_results", "excel_path", "mapping"
            ]}
        return {k: make_serializable(v, top_level=False) for k, v in obj.items()}
    
 
    elif isinstance(obj, list):
        return [make_serializable(v, top_level=False) for v in obj]
    
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    
    elif isinstance(obj, Path):
        return str(obj)
    elif hasattr(obj, 'item'):  
        return obj.item()
    elif hasattr(obj, 'isoformat'):  
        return obj.isoformat()
    
    return obj

# --- Nodi del grafo ---


# class LoadDocNode:
#     async def __call__(self, state: TestDesignState):
#         """Carica e processa il documento Word (estrazione paragrafi e header)"""
#         input_path = getattr(state, "docx_input_path", None) or os.path.join(
#             os.path.dirname(__file__),
#             "input",
#             "RU_Sportsbook_Platform_Fantacalcio_Prob. Form_v0.2 (1).docx"
#         )

#         paragraphs, headers = process_docx(input_path, os.path.dirname(input_path))
#         print(paragraphs[2])
#         return state.model_copy(update={
#             "input_path": input_path,
#             "paragraphs": paragraphs,
#             "headers": headers,
#         })


# class FilterDocNode:
#     async def __call__(self, state: TestDesignState):
#         """Filtra paragrafi e header in base a regole di esclusione (sommario, summary, ecc.)"""
#         #paragraphs = getattr(state, "paragraphs", [])
#         #headers = getattr(state, "headers", [])
#         paragraphs = state.paragraphs
#         headers = state.headers

#         print(f"PARAGRAPHS:  {paragraphs[3]}")
#         filtered_paragraphs = []
#         filtered_headers = []

#         for par, head in zip(paragraphs, headers):
#             if not head:
#                 continue

#             head_clean = head.strip().lower()
#             if (
#                 "== first line ==" in head_clean
#                 or "sommario" in head_clean
#                 or "summary" in head_clean
#             ):
#                 continue

#             filtered_paragraphs.append(par)
#             filtered_headers.append(head)

#         return state.model_copy(update={
#             "filtered_paragraphs": filtered_paragraphs,
#             "filtered_headers": filtered_headers,
#         })


# class CreateVectorStoreNode:
#     async def __call__(self, state: TestDesignState):
#         """Crea il VectorStore FAISS con gli embeddings dal documento"""
#         #input_path = getattr(state, "input_path", None)
#         input_path= state.input_path
#         # fallback se non è stato settato da LoadDocNode
#         if not input_path:
#             input_path = os.path.join(
#                 os.path.dirname(__file__),
#                 "input",
#                 "RU_Sportsbook_Platform_Fantacalcio_Prob. Form_v0.2 (1).docx"
#             )

#         rag_path = os.path.join(
#             os.path.dirname(__file__),
#             "input",
#             "RU_Sportsbook_Platform_Fantacalcio_Prob. Form_v0.2 (1).docx"
#         )

#         # Aggiungiamo un check di esistenza per evitare errori strani
#         if not os.path.exists(input_path):
#             raise FileNotFoundError(f"Il file DOCX non esiste: {input_path}")

#         chunks, _ = process_docx(input_path, os.path.dirname(rag_path))
#         #embedding_model = "text-embedding-3-large"
#         #embeddings = OpenAIEmbeddings(model=embedding_model)
#         #vectorstore = FAISS.from_texts(chunks, embeddings)
#         vectorstore=None
#         return state.model_copy(update={"vectorstore": vectorstore})



# class GenerateTCNode:
#     async def __call__(self, state: TestDesignState):
#         """Genera i test case con l'AI a partire dai paragrafi filtrati"""
#         # filtered_paragraphs = getattr(state, "filtered_paragraphs", [])
#         # filtered_headers = getattr(state, "filtered_headers", [])
#         # vectorstore = getattr(state, "vectorstore", None)
#         # mapping = extract_field_mapping()
#         filtered_paragraphs = state.filtered_paragraphs
#         filtered_headers = state.filtered_headers
#         vectorstore = state.vectorstore
#         mapping = extract_field_mapping()


#         print("=== GenerateTCNode.__call__ ===")
#         print(f"Numero paragrafi: {len(filtered_paragraphs)}")
#         print(f"Numero header: {len(filtered_headers)}")
#         #print(filtered_headers)
#         print("====================================================================================")
#         #print(filtered_paragraphs)
#         new_TC = await process_paragraphs(filtered_paragraphs, filtered_headers, vectorstore, mapping)
#         print("=== Risultato process_paragraphs ===")
#         print(new_TC)
#         #print(new_TC)
#         updated_json = merge_TC(new_TC)

#         # Aggiorna ID numerici sequenziali (TC-001, TC-002, ecc.)
#         start_number = 1
#         prefix = "TC"
#         padding = 3
#         for i, test_case in enumerate(updated_json["test_cases"], start=start_number):
#             old_id = test_case.get("ID", "N/A")
#             new_id = f"{prefix}-{str(i).zfill(padding)}"
#             test_case["ID"] = new_id
#         #print(updated_json)

#         return state.model_copy(update={"updated_json": updated_json})


# class WriteOutputNode:
#     async def __call__(self, state: TestDesignState):
#         """Salva i risultati su file JSON ed Excel"""
#         #print(f"UPDATED JSON: {getattr(state, 'updated_json', {})}")
#         #updated_json = getattr(state, "updated_json", {})
#         updated_json = state.updated_json
#         output_json_path = os.path.join(
#             os.path.dirname(__file__),
#             "outputs",
#             "generated_test_SportsBook_feedbackAI.json"
#         )
#         output_excel_path = os.path.join(
#             os.path.dirname(__file__),

#             "outputs",
#             "generated_test.xlsx"
#         )

#         os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

#         save_updated_json(updated_json, output_json_path)
#         convert_json_to_excel(updated_json, output_path=output_excel_path)

#         return state.model_copy(update={
#             "output_json_path": output_json_path,
#             "output_excel_path": output_excel_path,
#             "total_test_cases": len(updated_json.get("test_cases", [])),
#         })
    
# from fastapi import FastAPI, Form, HTTPException, File, UploadFile
# from typing import Optional
# import tempfile
# import shutil

# @app.post("/run")
# async def run_accredito_checker(    
#     docx_file: Optional[UploadFile] = File(None)
# ):

#     """
#     Esegue il flow per il documento 'Accredito Vincite Online':
#     - Estrae paragrafi e header dal docx
#     - Filtra contenuti irrilevanti
#     - Crea il vectorstore
#     - Genera i test case con LLM
#     - Salva in JSON ed Excel
    
#     Accepts three input methods:
#     1. docx_file: Upload a .docx file directly
#     2. docx_path: Provide a file path (server-side)
#     3. encrypted_state: Encrypted JSON state object
#     """

#     try:
#         initial_state = {}
#         temp_file_path = None

#         # Priority 1: File upload
#         if docx_file:
#             if not docx_file.filename.endswith('.docx'):
#                 raise HTTPException(
#                     status_code=400,
#                     detail="Il file deve essere in formato .docx"
#                 )
            
#             word_path = os.path.join("/tmp", docx_file.filename)
#             with open(word_path, "wb") as f:
#                 f.write(await docx_file.read())
#             # Add the *path* to the state
#             initial_state["docx_input_path"] = word_path

#     except HTTPException:
#         raise
#     except Exception as e:
#         raise HTTPException(
#             status_code=400,
#             detail=f"Errore nell'elaborazione dell'input: {str(e)}"
#         )

#     try:
#         # --- Crea il grafo ---
#         graph = StateGraph(TestDesignState)

#         graph.add_node("load_doc", LoadDocNode())
#         graph.add_node("filter_doc", FilterDocNode())
#         graph.add_node("create_vectorstore", CreateVectorStoreNode())
#         graph.add_node("generate_tc", GenerateTCNode())
#         graph.add_node("write_output", WriteOutputNode())

#         graph.set_entry_point("load_doc")
#         graph.add_edge("load_doc", "filter_doc")
#         graph.add_edge("filter_doc", "create_vectorstore")
#         graph.add_edge("create_vectorstore", "generate_tc")
#         graph.add_edge("generate_tc", "write_output")
#         graph.set_finish_point("write_output")

#         # --- Esegui ---
#         compiled = graph.compile()
#         result = await compiled.ainvoke(initial_state)

#         file_excel_path = result["output_excel_path"]
   

#         with open(file_excel_path,"rb") as f:
#             a = f.read()
        
#             file_excel = base64.b64encode(a)

#             decoded_excel = file_excel.decode("utf-8")
#         # safe_result = make_serializable(result)
#         # encrypted_result = encrypt_payload(safe_result)
#         if result["total_test_cases"]:
#             tot = result["total_test_cases"]
#         else:
#             tot = 0

#         return {"excel_base64":decoded_excel,"filename":os.path.basename(file_excel_path), "total_test_cases": tot}



#     except Exception as e:
#         raise HTTPException(
#             status_code=500,
#             detail=f"Errore durante l'esecuzione del workflow: {str(e)}"
#         )
    
#     finally:
#         # Clean up temp file if created
#         if temp_file_path and os.path.exists(temp_file_path):
#             try:
#                 os.remove(temp_file_path)
#             except Exception:
#                 pass  # Ignore cleanup errors




# --- Nodi del grafo ---
class RunTestDesignNode:
    async def __call__(self, state: TestDesignState):
        """Esegue la pipeline copertura requisiti su Excel e Word"""
        dictionary = state.dictionary 
        result = await run_pipeline(dictionary)
        
        if not isinstance(result, dict):
            print(f"[FATAL ERROR] run_pipeline ha restituito una stringa: {result}")
            raise TypeError(f"run_pipeline ha restituito un tipo inatteso ({type(result)}) invece di un dizionario. Contenuto: {result}")

# Ora result è sicuramente un dizionario, e puoi usare .get()
        print(f"[NODO] Excel path ricevuto: {result.get('excel_path')}") # OK 
        print(f"[NODO] Pipeline completata")
        print(f"[NODO] Excel path ricevuto: {result.get('excel_path')}")
        state.excel_path = result["excel_path"]
        #return state.model_copy(update={"excel_path": result["excel_path"]})
        return state

@app.post("/run")
async def run_test_design(
    
    # Part 1: The encrypted metadata (Note: name matches 'encrypted_state')
    # encrypted_state: str = Form(...), 
    encrypted_state: Optional[str] = Form(None),

    # Part 2: The optional files
    #docx_file: Optional[UploadFile] = File(None),
    image: Optional[list[UploadFile]] = None,
    text: Optional[UploadFile] = None,
    excel: Optional[UploadFile] = None
):
    """
    Esegue la pipeline di copertura tracciabilità
    """

    # 2. Save the uploaded files (if they exist) and get their paths
    initial_state =  {}
    # if excel_file:
    #     excel_path = os.path.join("/tmp", excel_file.filename) # Use a temp dir
    #     with open(excel_path, "wb") as f:
    #         f.write(await excel_file.read())
        # Add the *path* to the state so the pipeline can find it
        # initial_state["excel_input_path"] = excel_path
    dictionary= {}
    # if excel:
    #     dictionary["excel"] = excel
    # if text:
    #     dictionary["text"] = text
    # if image:
    #     dictionary["image"] = image

    # if dictionary:
    #     initial_state = TestDesignState(dictionary=dictionary)
    if excel:
        dictionary["excel"] = excel
        
    # 2. Gestione Testo (Docx/PDF) - CAMBIA LA CHIAVE IN "testo"
    if text:
        # PRIMA ERA: dictionary["text"] = text
        # CORREZIONE: Usa "testo" perché test_design.py controlla if tipi == {"testo", "excel"}
        dictionary["text"] = text 
        
    if image:
        dictionary["image"] = image

    # Controllo di sicurezza: Se mancano i file, ferma tutto subito
    if dictionary and "excel" in dictionary and "text" not in dictionary:
        raise HTTPException(status_code=400, detail="Hai caricato l'Excel ma manca il file di Testo (DocX/PDF).")

    if dictionary:
        initial_state = TestDesignState(dictionary=dictionary)

    # if docx_file:
    #     word_path = os.path.join("/tmp", docx_file.filename)
    #     with open(word_path, "wb") as f:
    #         f.write(await docx_file.read())
    #     # Add the *path* to the state
    #     initial_state["word_input_path"] = word_path

    
    # 3. Run your graph (it will now use the paths from initial_state)
    graph = StateGraph(TestDesignState)
    graph.add_node("run_test_design", RunTestDesignNode())
    graph.set_entry_point("run_test_design")
    graph.set_finish_point("run_test_design")
    
    compiled = graph.compile()
    result = await compiled.ainvoke(initial_state)
    print(f"Tipo di result: {type(result)}")
    print(f"Contenuto di result: {result}")
    if isinstance(result, dict):
        file_excel_path = result.get("excel_path")
    else:
        # Se è un oggetto Pydantic
        file_excel_path = getattr(result, "excel_path", None)
    if not file_excel_path:
        raise ValueError("excel_path non trovato nel risultato della pipeline")
    
    print(f"Excel path: {file_excel_path}")
    if not os.path.exists(file_excel_path):
        raise FileNotFoundError(f"File Excel non trovato: {file_excel_path}")

    with open(file_excel_path,"rb") as f:
        a = f.read()
    
        file_excel = base64.b64encode(a)

        decoded_excel = file_excel.decode("utf-8")
    # safe_result = make_serializable(result)
    # encrypted_result = encrypt_payload(safe_result)

    return {"excel_base64":decoded_excel,"filename":os.path.basename(file_excel_path)}
# --- Root endpoint ---
@app.get("/")
def home():
    return {"status": "ok", "message": "Copertura tracciabilità pronta"}
