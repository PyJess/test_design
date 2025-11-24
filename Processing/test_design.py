# from langchain.document_loaders import UnstructuredWordDocumentLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
# from langchain.vectorstores import FAISS
import pandas as pd
# from langchain_openai import ChatOpenAI
from docx import Document
import sys
import os
from typing import Dict, List, Tuple, Any
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import asyncio

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Input_extraction.extract_polarion_field_mapping import *
from utils.simple_functions import *
from utils.excel_fix_output import *
from llm.llm import *
import base64
#from Processing.controllo_sintattico import fill_excel_file

llm_client = LLMClient()

#embedding_model = "text-embedding-3-large"
PANDOC_EXE = "pandoc" 

def add_new_TC(new_TC, original_excel):

    new_TC_list = [tc for tc in new_TC_list if tc is not None]

    if not new_TC_list:
        print("Nessun nuovo TC da aggiungere (tutti i requirement sono già coperti)")
        return original_excel

    field_mapping = {
        'Title': 'Title',
        'Test Group': 'Test Group',
        'Channel': 'Canale',
        'Device': 'Dispositivo',
        'Priority': 'Priority',
        'Test Stage': 'Test Stage',
        'Reference System': 'Sistema di riferimento',
        'Preconditions': 'Precondizioni',
        'Execution Mode': 'Modalità Operativa',
        'Functionality': 'Funzionalità',
        'Test Type': 'Tipologia Test',
        'Dataset': 'Dataset',
        'Expected Result': 'Risultato Atteso',
        'Country': 'Country',
        'Type': 'Type',
        '_polarion': '_polarion'
    }

    all_columns = set()
    for test_data in original_excel.values():
        all_columns.update(test_data.keys())

    max_number = max(
        (int(test_data.get('#', 0)) for test_data in original_excel.values() 
         if '#' in test_data and isinstance(test_data['#'], (int, float))),
        default=0
    )

    for new_test in new_TC:
        test_id = new_test.get('ID', '')
        max_number += 1

        new_test_case = {}
        for col in all_columns:
            if col == 'Steps':
                new_test_case[col] = []
            else:
                new_test_case[col] = ''
        new_test_case['#'] = max_number

        for ai_field, json_field in field_mapping.items():
            if ai_field in new_test and new_test[ai_field] is not None and new_test[ai_field] != '':
                new_test_case[json_field] = new_test[ai_field]

        if 'Steps' in new_test and new_test['Steps']:
                new_test_case['Steps'] = new_test['Steps']
        original_excel[test_id] = new_test_case
    return original_excel

            
def save_updated_json(updated_json, output_path='updated_test_cases.json'):
    """
    Salva il JSON aggiornato su file.
    
    Args:
        updated_json (dict): JSON con i test cases aggiornati
        output_path (str): Percorso del file di output
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(updated_json, f, ensure_ascii=False, indent=2)
    print(f"JSON aggiornato salvato in: {output_path}")



async def prepare_prompt(input: Dict, context:str, mapping: str = None) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    """Prepare prompt for the LLM"""

    system_prompt = load_file(os.path.join(os.path.dirname(__file__), "..", "llm", "prompts",  "system_prompt.txt"))
    user_prompt = load_file(os.path.join(os.path.dirname(__file__), "..", "llm", "prompts",  "user_prompt.txt")) 
    schema = load_json(os.path.join(os.path.dirname(__file__), "..", "llm", "schema", "schema_output.json"))

    user_prompt = user_prompt.replace("{input}", json.dumps(input))
    mapping_as_string = mapping.to_json() 
    user_prompt = user_prompt.replace("{mapping}", mapping_as_string)

    context = "\n\n".join(context) if context else ""
    user_prompt= user_prompt.replace("{context}", context)

    print("finishing prepare prompt")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    return messages, schema

async def gen_TC(paragraph, context, mapping):
        print("=== gen_TC ===")
        print(f"Paragrafo: {paragraph[:200]}...")
        """Call LLM to generate test cases from paragraph"""
        paragraph = paragraph.page_content if hasattr(paragraph, 'page_content') else str(paragraph)
        messages, schema = await prepare_prompt(paragraph, context, mapping)
        print("starting calling llm")
        #print(f"{messages}")
        response = await llm_client.a_invoke_model(messages, schema)
        print("Test Case generato con successo!")
        print("Risposta dall'LLM:")
        print(response)
        return response


def create_vectordb(paragraph, vectorstore, k=3, similarity_threshold=0.75):
    docs_found = vectorstore.similarity_search_with_score(paragraph, k)

    closest_test=[]
    if docs_found:
        closest_doc, score = docs_found[0]
        score = 1 - (score / 2)
        if score >= similarity_threshold:  
            print(f"Score: {score}")
            closest_test.append(closest_doc.page_content)
        else:
            closest_test = None

    return closest_test
    

def merge_TC(new_TC):
    """
    Merge all the test cases in one json

    """
    all_test_cases = []
    
    for tc in new_TC:
        if tc is None:
            continue
            
        if isinstance(tc, list):
            all_test_cases.extend(tc)
        
        elif isinstance(tc, dict) and "test_cases" in tc:
            test_cases = tc["test_cases"]
            if isinstance(test_cases, list):
                all_test_cases.extend(test_cases)
            else:
                all_test_cases.append(test_cases)
        
        elif isinstance(tc, dict):
            all_test_cases.append(tc)
    
    return {
        "test_cases": all_test_cases,
        "total_count": len(all_test_cases)
    }

async def process_paragraphs(paragraphs, headers, vectorstore, mapping):
    """Process all paragraphs asynchronously to generate test cases."""
    
    async def process_single_paragraph(i, par):
        print(f"numero: {i}")
        print(f"\n--- Paragrafo {i}/{len(paragraphs)} ---")
        #print(f"\n--- Paragrafo {i}/{len(paragraphs)} ---")
        print(f"Contenuto paragrafo: {str(par)[:200]}...")  # stampa i primi 200 caratteri

        #context = create_vectordb(par, vectorstore, k=3, similarity_threshold=0.75)
        #print(f"Context retrieved: {context}")
        print("Preparazione chiamata LLM")

        context = ""
        while True:
            tc = await gen_TC(par, context, mapping)

            if isinstance(tc, dict) and "test_cases" in tc:
                for test_case in tc["test_cases"]:
                    test_case["_polarion"] = headers[i - 1] 

                print("Output LLM ricevuto:")
                print(tc)
                return tc

    # Crea tutte le tasks e le esegue in parallelo
    tasks = [process_single_paragraph(i, par) for i, par in enumerate(paragraphs, 1)]
    new_TC = await asyncio.gather(*tasks)
    
    return new_TC


def encode_image(image_path: str) -> str:
    """
    Encode an image file as a base64 string.

        Parameters:
            image_path (str): The file path to the image to be encoded

        Returns:
            str: a base64 encoded string representation of the image
    """
    try:
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
            if not image_data:
                print(f"Il file {image_path} è vuoto")
                return None
            return base64.b64encode(image_data).decode('utf-8')
    except FileNotFoundError as e:
        print(f"Immagine  {image_path} non trovata: {e}")
        return None
    except IOError as e:
        print(f"Errore nella lettura dell'immagine {image_path}: {e}")
        return None



async def main():

    input_path= os.path.join(os.path.dirname(__file__), "..", "input","ru_accredito_vincite_online_v0.2.docx")
    print(os.path.dirname(input_path))
    paragraphs, headers =process_docx(input_path, os.path.dirname(input_path))

    filtered_paragraphs = []
    filtered_headers = []

    for par, head in zip(paragraphs, headers):
        if not head:
            continue

        head_clean = head.strip().lower()

        if (
            "== first line ==" in head_clean
            or "sommario" in head_clean or "summary" in head_clean
        ):
            continue  

        filtered_paragraphs.append(par)
        filtered_headers.append(head)

    rag_path=os.path.join(os.path.dirname(__file__), "..", "input", "ru_accredito_vincite_online_v0.2.docx")
    chunks, _ = process_docx(input_path, os.path.dirname(rag_path))
    #embeddings = OpenAIEmbeddings(model=embedding_model)
    #vectorstore = FAISS.from_texts(chunks, embeddings)
    vectorstore=None

    mapping = extract_field_mapping()
    #print("finishing mapping")
    new_TC= await process_paragraphs(filtered_paragraphs, filtered_headers, vectorstore, mapping)

    updated_json=merge_TC(new_TC)

    print("Finishing generating test cases from LLM")

    start_number = 1
    prefix = "TC"
    padding = 3
    for i, test_case in enumerate(updated_json["test_cases"], start=start_number):
        old_id = test_case.get("ID", "N/A")
        new_id = f"{prefix}-{str(i).zfill(padding)}"
        test_case["ID"] = new_id
    
    print(f"\n Total test cases updated: {len(updated_json['test_cases'])}")

    output_path= os.path.join(os.path.dirname(__file__), "..", "outputs", "generated_test_AccreditoVincite_feedbackAI.json")
    save_updated_json(updated_json, output_path)
    convert_json_to_excel(updated_json, output_path=os.path.join(os.path.dirname(__file__), "..", "outputs", "generated_test_AccreditoVincite_feedbackAI.xlsx"))


# if __name__ == "__main__":
#     asyncio.run(main())

async def run_pipeline(input_word_path: str):
    print(f"Avvio pipeline generazione test case su {input_word_path}")

    word_path = Path(input_word_path)
    if not word_path.exists():
        raise FileNotFoundError(f"Word non trovato: {word_path}")
    
    if word_path.endswith("pdf"):
        print("ciao")

    if word_path.endswith("docx"):

        # Elaborazione Word
        paragraphs, headers = process_docx(word_path, word_path.parent)

        # Filtraggio intestazioni
        filtered_paragraphs = []
        filtered_headers = []
        for par, head in zip(paragraphs, headers):
            if not head:
                continue
            head_clean = head.strip().lower()
            if "== first line ==" in head_clean or "sommario" in head_clean or "summary" in head_clean or "introduzione" in head_clean or "introduction" in head_clean:
                continue
            filtered_paragraphs.append(par)
            filtered_headers.append(head)

        # Preparazione RAG / chunks
        chunks, _ = process_docx(word_path, word_path.parent)
        vectorstore = None  # eventualmente implementa embeddings/FAISS

        # Mapping campi
        mapping = extract_field_mapping()

        # Generazione nuovi test case
        new_TC = await process_paragraphs(filtered_paragraphs, filtered_headers, vectorstore, mapping)
        updated_json = merge_TC(new_TC)
        print("Generazione test case completata tramite LLM")

        # Assegnazione ID
        start_number = 1
        prefix = "TC"
        padding = 3
        for i, test_case in enumerate(updated_json["test_cases"], start=start_number):
            test_case["ID"] = f"{prefix}-{str(i).zfill(padding)}"

        print(f"Totale test case aggiornati: {len(updated_json['test_cases'])}")

        # Salvataggio JSON e Excel
        output_dir = word_path.parent.parent / "outputs"
        output_dir.mkdir(exist_ok=True)
        output_json_path = output_dir / f"{word_path.stem}_feedbackAI.json"
        output_excel_path = output_dir / f"{word_path.stem}_feedbackAI.xlsx"

        save_updated_json(updated_json, output_json_path)
        convert_json_to_excel(updated_json, output_excel_path)

        #print(f"File salvati: \nJSON -> {output_json_path}\nExcel -> {output_excel_path}")

        output=fix_labels_with_order(output_excel_path)

        return {"status": "ok", "json_path": str(output_json_path), "excel_path": str(output),
                "total_cases": len(updated_json["test_cases"])}


if __name__ == "__main__":
    sample_word = os.path.join(os.path.dirname(__file__), "..", "input", "RU_Sportsbook_Platform_Fantacalcio_Prob. Form_v0.2 (1).docx")
    asyncio.run(run_pipeline(sample_word))







