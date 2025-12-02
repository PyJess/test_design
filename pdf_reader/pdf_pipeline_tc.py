import asyncio
import json
from pathlib import Path
from typing import List, Dict, Any
import os
from .pdf_data_extraction import agent_data_extraction
from dotenv import load_dotenv

load_dotenv()
BASE_DIR = Path(__file__).parent 
OUTPUT_JSON_FILENAME = "regolamento_strutturato.json"



def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
    
def process_json(json_path:Path):
    json_file = load_json(json_path)

    sezioni = json_file.get("sezioni", [])
    titoli = []
    contenuti = []
    for sezione in sezioni:
        titolo = sezione.get("titolo")
        contenuto = sezione.get("contenuto")
        
        titoli.append(titolo)
        contenuti.append(contenuto)
        # print(f"**{titolo}:**\n{contenuto}\n---")
        
    return titoli,contenuti
    
async def run_extraction_and_retrieval_pipeline(
    pdf_path: Path, 
    force_reprocess: bool = False,
): 
    
    output_json_path = BASE_DIR / OUTPUT_JSON_FILENAME
    parsed_data = None

    # --- Tentativo di caricamento file esistente ---
    if output_json_path.exists() and not force_reprocess:
        try:
            parsed_data = load_json(output_json_path)
        except json.JSONDecodeError:
            print("Avviso: Impossibile caricare il JSON esistente. Forzo la rielaborazione")
            parsed_data = None
            force_reprocess = True 

    # --- Fase 1: Estrazione o Caricamento ---
    if force_reprocess or parsed_data is None:
        print("Fase 1: Estrazione del documento in JSON (con LLM) in corso...")
        parsed_data = await agent_data_extraction(
            file_path=pdf_path,
            output_json_path=output_json_path,
        )
    else:
        print("Fase 1: File JSON strutturato esistente, Estrazione saltata")

    # --- Verifica Estrazione ---
    if not parsed_data or "sezioni" not in parsed_data:
        print("Estrazione fallita o dati non trovati.")
        return [], []
        
    # --- Processo del JSON ---
    print("Fase 2: Processamento del JSON per estrarre titoli e contenuti")
    titoli, contenuti = process_json(output_json_path)

    return titoli, contenuti

# test = Path(__file__).parent/"20210930_REGOLAMENTO_SVT_2021.pdf"
# asyncio.run(run_extraction_and_retrieval_pipeline(test))