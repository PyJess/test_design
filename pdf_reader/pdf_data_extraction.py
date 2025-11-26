from llm_pdf import LLMClient
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
import fitz  
from docx import Document
from dotenv import load_dotenv
from pathlib import Path
from langchain.prompts import PromptTemplate
import os
import json 
import asyncio

load_dotenv()
llm_client = LLMClient()

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

# pdf_data_extraction.py (RIADATTATO)
# ... (import, read_document, response_schemas, output_parser, format_instructions rimangono gli stessi)

async def agent_data_extraction(file_path:str, output_json_path:str):
    print(f"[INFO] Lettura file: {file_path}")
    text = read_document(file_path)
    text = text[:18000]

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

    messages = [
        {"role": "system", "content": "Sei un assistente esperto nella strutturazione di documenti in sezioni. L'output DEVE essere un JSON valido conforme allo schema richiesto."},
        {"role": "user", "content": formatted_prompt}
    ]

    print("[INFO] Analisi con LLM in corso...")

    response_content = await llm_client.a_invoke_model(messages) 

    # --- BLOCCO PARSING CORRETTO ---
    try:
        parsed_data = output_parser.parse(response_content)
    except Exception as e:
        print(f"Errore nel parsing con StructuredOutputParser, provo json.loads: {e}")
        try:
            parsed_data = json.loads(response_content)
        except Exception as e:
            print(f"Errore critico nel parsing JSON: {e}")
            raise
    # --- FINE BLOCCO PARSING CORRETTO ---

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(parsed_data, f, ensure_ascii=False, indent=2)
    print(f"File JSON salvato in: {output_json_path}")

    return parsed_data

BASE_DIR = Path(r"C:\Users\BencichWilliam\Desktop\test_design_xhesina\test_design\pdf_reader")

OUTPUT_FILENAME = "regolamento_strutturato.json" 

output_json_path = BASE_DIR / OUTPUT_FILENAME

pdf_test = r"C:\Users\BencichWilliam\Desktop\test_design_xhesina\test_design\pdf_reader\20210930_REGOLAMENTO_SVT_2021.pdf"


asyncio.run(agent_data_extraction(str(pdf_test), str(output_json_path)))