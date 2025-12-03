# from langchain.document_loaders import UnstructuredWordDocumentLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
# from langchain.vectorstores import FAISS
import pandas as pd
import shutil
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
from pdf_reader.pdf_pipeline_tc import run_extraction_and_retrieval_pipeline

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
        # print(f"{messages}")
        response = await llm_client.a_invoke_model(messages, schema)
        print("Test Case generato con successo!")
        print("Risposta dall'LLM:")
        #print(response)
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

def fix_json(json_str):
        """Prova a riparare errori comuni nel JSON generato dall'LLM."""
        # Rimuove virgole finali prima di chiusura di oggetti o array
        json_str = re.sub(r',(\s*[\]}])', r'\1', json_str)
        # Aggiunge eventuali virgole mancanti tra proprietà (semplificato)
        json_str = re.sub(r'(".*?")\s*("\w+"\s*:)', r'\1,\2', json_str)
        return json_str


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

        max_attempts = 8
        attempts = 0
        context = ""
        while attempts < max_attempts:
            attempts += 1
            try:
                print(f"Generazione nuovo TC, tentativo {attempts}")
                tc = await gen_TC(par, context, mapping)
            except Exception as e:
                print(f"Errore durante la chiamata LLM al tentativo {attempts}: {e}")
                tc = None
            if isinstance(tc, dict) and "test_cases" in tc:
                for test_case in tc["test_cases"]:
                    test_case["_polarion"] = headers[i - 1] 


                tc_str = json.dumps(tc)  # converte dict in stringa
                tc_fixed_str = fix_json(tc_str)  # rimuove virgole finali
                #tc = json.loads(tc_str)
                tc = json.loads(tc_fixed_str)
                print("Output LLM ricevuto:")
                #print(tc)
                return tc

    # Crea tutte le tasks e le esegue in parallelo
    tasks = [process_single_paragraph(i, par) for i, par in enumerate(paragraphs, 1)]
    new_TC = await asyncio.gather(*tasks)
    
    return new_TC


async def gen_TC_with_image(paragraph, context, mapping):
        print("=== gen_TC ===")
        print(f"Paragrafo: {paragraph[:200]}...")
        """Call LLM to generate test cases from paragraph"""
        paragraph = paragraph.page_content if hasattr(paragraph, 'page_content') else str(paragraph)
        #messages, schema = await prepare_prompt(paragraph, context, mapping)
        print("starting calling llm")
        #print(f"{messages}")
        print("prepare prompts with images")
        system_prompt2 = load_file(os.path.join(os.path.dirname(__file__), "..", "llm", "prompts",  "system_prompt.txt"))
        user_prompt = load_file(os.path.join(os.path.dirname(__file__), "..", "llm", "prompts",  "user_prompt.txt"))
        user_prompt = user_prompt.replace("{input}", json.dumps(paragraph)) 
        #schema = load_json(os.path.join(os.path.dirname(__file__), "..", "llm", "schema", "schema_output.json"))
        response = await llm_client.process_images_from_folder(system_prompt2, "image", user_prompt)
        #response = await llm_client.a_invoke_model(messages, schema)
        print("Test Case generato con successo!")
        #print("Risposta dall'LLM:")
        #print(response)
        return response


async def process_paragraphs_with_image(paragraphs, headers, vectorstore, mapping):
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
        max_attempts = 5
        attempts = 0
        while attempts < max_attempts:
            attempts += 1
            print(f"Generazione nuovo TC, tentativo {attempts}")
            tc = await gen_TC_with_image(par, context, mapping)
            #rint(f"[TEST CASES] {tc}")
            if isinstance(tc, str):
                tc = json.loads(tc)
            if isinstance(tc, dict) and "test_cases" in tc:
                for test_case in tc["test_cases"]:
                    test_case["_polarion"] = headers[i - 1] 
                return tc
            else:

                print(f"⚠️ Output non valido per il paragrafo {i}: {tc}")
                return {"test_cases": []}

    # Crea tutte le tasks e le esegue in parallelo
    print("process single paragraf")
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

async def run_pipeline(dizionario: dict):
    tipi=set()
    for key in list(dizionario.keys()):
        #value = dizionario[key]
        if key== "testo" or key== "text" or key== "TEXT":
            tipi.add("testo")
        if key=="image":
            tipi.add("image")
        if key=="excel":
            tipi.add("excel")
        
    if tipi =={"testo"}:
            print("only text")
            for key, value in dizionario.items():
                docx_file=value
                
                word_path = os.path.join("/tmp", docx_file.filename)
                # word_path = os.path.join("", docx_file.filename)
                
                with open(word_path, "wb") as f:
                    f.write(await docx_file.read())

                word_path = Path(word_path)
                if not word_path.exists():
                    raise FileNotFoundError(f"Word non trovato: {word_path}")
                
                if word_path.suffix == ".pdf":
                    titles,contents_of_title = await run_extraction_and_retrieval_pipeline(word_path)
                    
                    mapping = extract_field_mapping()
                    context = ""
                    
                    # all_pdf_test_cases = []
                    MAX_ATTEMPTS = 5
                    
                    async def process_single_section(title, content, context, mapping, max_attempts):
                        """Esegue gen_TC con logica di retry per una singola sezione."""
                        title_clean = title.strip().lower()
                        
                        attempts = 0
                        while attempts < max_attempts:
                            attempts += 1
                            print(f"-> Tentativo {attempts}/{max_attempts} per la sezione: {title_clean[:50]}...")
                            
                            try:
                                # 1. CHIAMATA ALL'LLM
                                new_TC = await gen_TC(content, context, mapping)
                                
                                # 2. CONVERSIONE E PARSING
                                if isinstance(new_TC, str):
                                    new_TC = json.loads(new_TC)
                                
                                # 3. VERIFICA E SUCCESS
                                if isinstance(new_TC, dict) and "test_cases" in new_TC:
                                    # Ritorna i risultati e il titolo (necessario per l'accumulo)
                                    return new_TC["test_cases"], title 
                                else:
                                    print(f"Output LLM valido, ma manca la chiave 'test_cases'. Riprovo.")

                            except json.JSONDecodeError:
                                print(f"Tentativo {attempts} fallito: Errore di parsing JSON. Riprovo.")
                                
                            except Exception as e:
                                print(f"Tentativo {attempts} fallito: Errore generico: {e}. Riprovo.")
                                
                        # Fallimento permanente
                        print(f"Fallimento permanente dopo {max_attempts} tentativi per la sezione: {title_clean}")
                        return [], title 

                    # 2. CREAZIONE DELLA LISTA DI TASK (COROUTINE)
                    tasks = []
                    
                    for title, content in zip(titles, contents_of_title):
                        # Filtro: escludi intestazioni/sommari, ecc.
                        if not title:
                            continue
                        title_clean = title.strip().lower()
                        if "== first line ==" in title_clean or "sommario" in title_clean or "summary" in title_clean or "introduzione" in title_clean or "introduction" in title_clean:
                            continue
                        
                        # Aggiungi il task alla lista
                        # Qui la funzione process_single_section viene CHIAMATA, e restituisce un oggetto coroutine.
                        task = process_single_section(title, content, context, mapping, MAX_ATTEMPTS)
                        tasks.append(task)
                        
                    # 3. ESECUZIONE PARALLELA
                    all_results = await asyncio.gather(*tasks)
                    
                    # 4. ACCUMULO FINALE 
                    all_pdf_test_cases = []
                    
                    for test_cases_list, source_title in all_results:
                        # Aggiungi il campo _polarion e accumula
                        for tc in test_cases_list:
                            tc["_polarion"] = source_title
                        all_pdf_test_cases.extend(test_cases_list)
                        
                    updated_json = {
                        "test_cases": all_pdf_test_cases,
                        "total_count": len(all_pdf_test_cases)
                    }
                    #     try:
                    #         new_TC = await gen_TC(content, context, mapping)
                            
                    #         if isinstance(new_TC, str):
                    #             new_TC = json.loads(new_TC) 
                                
                    #         if isinstance(new_TC, dict) and "test_cases" in new_TC:
                    #             for tc in new_TC["test_cases"]:
                    #                 tc["_polarion"] = title 
                    #             all_pdf_test_cases.extend(new_TC["test_cases"])
                                
                    #     except Exception as e:
                    #         print(f"Errore generazione TC per {title}: {e}")
                        
                    # updated_json = {
                    #         "test_cases": all_pdf_test_cases,
                    #         "total_count": len(all_pdf_test_cases)
                    #     }
              

                    # Assegnazione ID
                    start_number = 1
                    prefix = "TC"
                    padding = 3
                    for i, test_case in enumerate(updated_json["test_cases"], start=start_number):
                        test_case["ID"] = f"{prefix}-{str(i).zfill(padding)}"

                    print(f"Totale test case aggiornati: {len(updated_json['test_cases'])}")

                    # Salvataggio JSON e Excel
                    #output_dir = word_path.parent.parent / "outputs"
                    output_dir = Path(__file__).parent.parent / "outputs"
                    output_dir.mkdir(exist_ok=True)
                    output_json_path = output_dir / f"{word_path.stem}_feedbackAI.json"
                    output_excel_path = output_dir / f"{word_path.stem}_feedbackAI.xlsx"

                    save_updated_json(updated_json, output_json_path)
                    convert_json_to_excel(updated_json, output_excel_path)

                    #print(f"File salvati: \nJSON -> {output_json_path}\nExcel -> {output_excel_path}")

                    output=fix_labels_with_order(output_excel_path)
                    output_excel_path.unlink() 

                    return {"status": "ok", "json_path": str(output_json_path), "excel_path": str(output),
                            "total_cases": len(updated_json["test_cases"])}
                if word_path.suffix == ".docx":

                    # Elaborazione Word
                    paragraphs, headers = process_docx(word_path, word_path.parent)
                    #paragraphs, headers, images_dict, images_per_chunk = process_docx_with_image(word_path, word_path.parent)
                    # Filtraggio intestazioni
                    filtered_paragraphs = []
                    filtered_headers = []
                    #filtered_images = []
                    #for idx, (par, head) in enumerate(zip(paragraphs, headers)):
                    for par, head in zip(paragraphs, headers):
                        if not head:
                            continue
                        head_clean = head.strip().lower()
                        if "== first line ==" in head_clean or "sommario" in head_clean or "summary" in head_clean or "introduzione" in head_clean or "introduction" in head_clean:
                            continue
                        filtered_paragraphs.append(par)
                        filtered_headers.append(head)
                        #filtered_images.append(images_per_chunk[idx] if idx < len(images_per_chunk) else [])
                    
                    # Preparazione RAG / chunks
                    chunks, _ = process_docx(word_path, word_path.parent)
                    vectorstore = None  # eventualmente implementa embeddings/FAISS

                    # Mapping campi
                    mapping = extract_field_mapping()

                    # Generazione nuovi test case
                    #new_TC = await process_paragraphs(filtered_paragraphs, filtered_headers, vectorstore, mapping, images_dict, filtred_images)
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
                    #output_dir = word_path.parent.parent / "outputs"
                    output_dir = Path(__file__).parent.parent / "outputs"
                    output_dir.mkdir(exist_ok=True)
                    output_json_path = output_dir / f"{word_path.stem}_feedbackAI.json"
                    output_excel_path = output_dir / f"{word_path.stem}_feedbackAI.xlsx"

                    save_updated_json(updated_json, output_json_path)
                    convert_json_to_excel(updated_json, output_excel_path)

                    #print(f"File salvati: \nJSON -> {output_json_path}\nExcel -> {output_excel_path}")

                    output=fix_labels_with_order(output_excel_path)
                    output_excel_path.unlink() 

                    return {"status": "ok", "json_path": str(output_json_path), "excel_path": str(output),
                            "total_cases": len(updated_json["test_cases"])}
                

    if tipi== {"image"}:
            print("only image")
            os.makedirs("image", exist_ok=True)

            for file in os.listdir("image"):
                file_path = os.path.join("image", file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            i=0
            for key, value_list in dizionario.items():
                for i, image_file in enumerate(value_list, start=1):
                    word_path = os.path.join("/tmp", image_file.filename)
                    with open(word_path, "wb") as f:
                        f.write(await image_file.read())
                    input_image_path = Path(word_path)
                    i+=1

                    filename = f"_{i}" + os.path.basename(input_image_path) 
                    dest_path = os.path.join("image", filename)
                    shutil.copy(input_image_path, dest_path)
                    print(f"Salvata immagine: {dest_path}")
            
            for root, dirs, files in os.walk("image"):  
                for file in files:
                    full_path = os.path.join(root, file)
                    if file.lower().endswith("docx"):
                        images_path= estrai_immagini_da_docx(full_path)
                    elif file.lower().endswith("pdf"):
                        images_path= estrai_immagini_da_pdf(full_path)

            system_prompt=load_file(os.path.join(os.path.dirname(__file__), "..", "llm", "prompts", "Figma", "system_prompt1.txt"))
            response= await llm_client.process_images_from_folder(system_prompt, "image")

            system_prompt2= load_file(os.path.join(os.path.dirname(__file__), "..", "llm", "prompts", "system_prompt.txt"))
            user_prompt=load_file(os.path.join(os.path.dirname(__file__), "..", "llm", "prompts", "Figma", "user_prompt.txt")) 
            user_prompt=user_prompt.replace("{flow}", response)

            while True:
                tc = await llm_client.process_images_from_folder(system_prompt2, "image", user_prompt)
                print(f"[OUTPUT]: {tc}")
                tc = json.loads(tc)
                if isinstance(tc, dict) and "test_cases" in tc:
                    for test_case in tc["test_cases"]:
                        test_case["_polarion"] = "Image from Figma"

                    print("Output LLM ricevuto:")
                    # print(tc)
                    break

            start_number = 1
            prefix = "TC"
            padding = 3
            for i, test_case in enumerate(tc["test_cases"], start=start_number):
                test_case["ID"] = f"{prefix}-{str(i).zfill(padding)}"

            print(f"Totale test case aggiornati: {len(tc['test_cases'])}")

            # Salvataggio JSON e Excel
            output_dir = "outputs"
            os.makedirs(output_dir, exist_ok=True)
            output_json_path = os.path.join(output_dir, "image_feedbackAI.json")
            output_excel_path = os.path.join(output_dir, "image_feedbackAI.xlsx")

            save_updated_json(tc, output_json_path)
            convert_json_to_excel(tc, output_excel_path)

            #print(f"File salvati: \nJSON -> {output_json_path}\nExcel -> {output_excel_path}")

            output=fix_labels_with_order(output_excel_path)
            Path(output_excel_path).unlink(missing_ok=True)

            return {"status": "ok", "json_path": str(output_json_path), "excel_path": str(output),
                    "total_cases": len(tc["test_cases"])}


    if tipi == {"testo", "excel"}:
        print("need to be implemented")

    if tipi == {"testo", "image"}:
        print("text and image")

        i=0
        for key, value_list in dizionario.items():
                if key== "image":
                    os.makedirs("image", exist_ok=True)
                    for file in os.listdir("image"):
                        file_path = os.path.join("image", file)
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                
                    for i, image_file in enumerate(value_list, start=1):
                        word_path = os.path.join("/tmp", image_file.filename)
                        with open(word_path, "wb") as f:
                            f.write(await image_file.read())
                        input_image_path = Path(word_path)
                        i+=1

                        filename = f"_{i}" + os.path.basename(input_image_path) 
                        dest_path = os.path.join("image", filename)
                        shutil.copy(input_image_path, dest_path)
                        print(f"Salvata immagine: {dest_path}")
                    #print(value)
                    #salvo tutte le immagini nella specifica folder per poter pocessare llm con immagine


                    for root, dirs, files in os.walk("image"):  
                        for file in files:
                            full_path = os.path.join(root, file)
                            if file.lower().endswith("docx"):
                                images_path= estrai_immagini_da_docx(full_path)
                            elif file.lower().endswith("pdf"):
                                images_path= estrai_immagini_da_pdf(full_path)

        #una volta salvate le immagini processo il testo
        for key, value in dizionario.items():
            if key== "testo" or key== "text" or key== "TEXT":
                docx_file=value
                    
                word_path = os.path.join("/tmp", docx_file.filename)
                with open(word_path, "wb") as f:
                    f.write(await docx_file.read())

                word_path = Path(word_path)
                if not word_path.exists():
                    raise FileNotFoundError(f"Word non trovato: {word_path}")
                    
                #print(value)
                if word_path.suffix.lower() == ".pdf":
                    titles,contents_of_title = await run_extraction_and_retrieval_pipeline(word_path)
                    
                    mapping = extract_field_mapping()
                    context = ""
                    
                    # all_pdf_test_cases = []
                    MAX_ATTEMPTS = 5
                    
                    async def process_single_section(title, content, context, mapping, max_attempts):
                        """Esegue gen_TC con logica di retry per una singola sezione."""
                        title_clean = title.strip().lower()
                        
                        attempts = 0
                        while attempts < max_attempts:
                            attempts += 1
                            print(f"-> Tentativo {attempts}/{max_attempts} per la sezione: {title_clean[:50]}...")
                            
                            try:
                                # 1. CHIAMATA ALL'LLM
                                new_TC = await gen_TC(content, context, mapping)
                                
                                # 2. CONVERSIONE E PARSING
                                if isinstance(new_TC, str):
                                    new_TC = json.loads(new_TC)
                                
                                # 3. VERIFICA E SUCCESS
                                if isinstance(new_TC, dict) and "test_cases" in new_TC:
                                    # Ritorna i risultati e il titolo (necessario per l'accumulo)
                                    return new_TC["test_cases"], title 
                                else:
                                    print(f"Output LLM valido, ma manca la chiave 'test_cases'. Riprovo.")

                            except json.JSONDecodeError:
                                print(f"Tentativo {attempts} fallito: Errore di parsing JSON. Riprovo.")
                                
                            except Exception as e:
                                print(f"Tentativo {attempts} fallito: Errore generico: {e}. Riprovo.")
                                
                        # Fallimento permanente
                        print(f"Fallimento permanente dopo {max_attempts} tentativi per la sezione: {title_clean}")
                        return [], title 

                    # 2. CREAZIONE DELLA LISTA DI TASK (COROUTINE)
                    tasks = []
                    
                    for title, content in zip(titles, contents_of_title):
                        # Filtro: escludi intestazioni/sommari, ecc.
                        if not title or not content:
                            continue
                        title_clean = title.strip().lower()
                        if "== first line ==" in title_clean or "sommario" in title_clean or "summary" in title_clean or "introduzione" in title_clean or "introduction" in title_clean:
                            continue
                        
                        # Aggiungi il task alla lista
                        # Qui la funzione process_single_section viene CHIAMATA, e restituisce un oggetto coroutine.
                        task = process_single_section(title, content, context, mapping, MAX_ATTEMPTS)
                        tasks.append(task)
                        
                    # 3. ESECUZIONE PARALLELA
                    all_results = await asyncio.gather(*tasks)
                    
                    # 4. ACCUMULO FINALE 
                    all_pdf_test_cases = []
                    
                    for test_cases_list, source_title in all_results:
                        # Aggiungi il campo _polarion e accumula
                        for tc in test_cases_list:
                            tc["_polarion"] = source_title
                        all_pdf_test_cases.extend(test_cases_list)
                        
                    updated_json = {
                        "test_cases": all_pdf_test_cases,
                        "total_count": len(all_pdf_test_cases)
                    }
                    # Assegnazione ID
                    start_number = 1
                    prefix = "TC"
                    padding = 3
                    for i, test_case in enumerate(updated_json["test_cases"], start=start_number):
                        test_case["ID"] = f"{prefix}-{str(i).zfill(padding)}"

                    print(f"Totale test case aggiornati: {len(updated_json['test_cases'])}")

                    # Salvataggio JSON e Excel
                    #output_dir = word_path.parent.parent / "outputs"
                    output_dir = Path(__file__).parent.parent / "outputs"
                    output_dir.mkdir(exist_ok=True)
                    output_json_path = output_dir / f"{word_path.stem}_feedbackAI.json"
                    output_excel_path = output_dir / f"{word_path.stem}_feedbackAI.xlsx"

                    save_updated_json(updated_json, output_json_path)
                    convert_json_to_excel(updated_json, output_excel_path)

                    #print(f"File salvati: \nJSON -> {output_json_path}\nExcel -> {output_excel_path}")

                    output=fix_labels_with_order(output_excel_path)
                    output_excel_path.unlink() 

                    return {"status": "ok", "json_path": str(output_json_path), "excel_path": str(output),
                            "total_cases": len(updated_json["test_cases"])}

                elif word_path.suffix.lower() == ".docx":

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
                    print("process paragraph with image")
                    new_TC = await process_paragraphs_with_image(filtered_paragraphs, filtered_headers, vectorstore, mapping)
                    updated_json = merge_TC(new_TC)
                    print("Generazione test case completata tramite LLM")

                    # Assegnazione ID
                    start_number = 1
                    prefix = "TC"
                    padding = 3
                    for i, test_case in enumerate(updated_json["test_cases"], start=start_number):
                        test_case["ID"] = f"{prefix}-{str(i).zfill(padding)}"
                        test_case["Dataset"] = ""

                    print(f"Totale test case aggiornati: {len(updated_json['test_cases'])}")

                    # Salvataggio JSON e Excel
                    #output_dir = word_path.parent.parent / "outputs"
                    output_dir = Path(__file__).parent.parent / "outputs"
                    output_dir.mkdir(exist_ok=True)
                    output_json_path = output_dir / f"{word_path.stem}_feedbackAI.json"
                    output_excel_path = output_dir / f"{word_path.stem}_feedbackAI.xlsx"

                    save_updated_json(updated_json, output_json_path)
                    convert_json_to_excel(updated_json, output_excel_path)

                    #print(f"File salvati: \nJSON -> {output_json_path}\nExcel -> {output_excel_path}")

                    output=fix_labels_with_order(output_excel_path)
                    Path(output_excel_path).unlink(missing_ok=True)

                    return {"status": "ok", "json_path": str(output_json_path), "excel_path": str(output),
                            "total_cases": len(updated_json["test_cases"])}


    return f"combinazione non gestita: {tipi}"

if __name__ == "__main__":
    #sample_word = os.path.join(os.path.dirname(__file__), "..", "input", "RU_Sportsbook_Platform_Fantacalcio_Prob. Form_v0.2 (1).docx")
    #sample_word= r"c:\Users\x.hita\Downloads\PRJ0015694_AI Angel Numera_Analisi Funzionale_DDA_v02.docx"
    sample_pdf= r"C:\Users\x.hita\Downloads\Screenshot 2025-11-25 151542.png"
    sample=r"C:\Users\x.hita\Downloads\Anagrafica Spagna\Anagrafica Spagna\PRJ0014382 SF - SEU Spagna Anagrafica v1.3.docx"
    # img1=r"C:\Users\x.hita\Downloads\UX_UI App SEVV - Agile II (14)\✅ Tabellone\Tabellone\SVT.png"
    # img2=r"C:\Users\x.hita\Downloads\UX_UI App SEVV - Agile II (14)\✅ Tabellone\Tabellone\SVT-1.png"
    # img3=r"C:\Users\x.hita\Downloads\UX_UI App SEVV - Agile II (14)\✅ Tabellone\Tabellone\SVT-2.png"
    # img4=r"C:\Users\x.hita\Downloads\UX_UI App SEVV - Agile II (14)\✅ Tabellone\Tabellone\SVT-3.png"
    # img5=r"C:\Users\x.hita\Downloads\UX_UI App SEVV - Agile II (14)\✅ Tabellone\Tabellone\SVT-4.png"
    # img6=r"C:\Users\x.hita\Downloads\UX_UI App SEVV - Agile II (14)\✅ Tabellone\Tabellone\SVT-5.png" 
    # img7= r"C:\Users\x.hita\Downloads\UX_UI App SEVV - Agile II (14)\✅ Tabellone\Schedina\SVT.png"
    # img8=r"C:\Users\x.hita\Downloads\UX_UI App SEVV - Agile II (14)\✅ Tabellone\Schedina\SVT-1.png"
    # img9=r"C:\Users\x.hita\Downloads\UX_UI App SEVV - Agile II (14)\✅ Tabellone\Schedina\SVT-2.png"
    # img10= r"C:\Users\x.hita\Downloads\UX_UI App SEVV - Agile II (14)\✅ Tabellone\Schedina\SVT-3.png"
    #sample_image= r"C:\Users\x.hita\Downloads\Doc2.docx"
    #dzionario={ img1: "image", img2: "image", img3: "image", img4: "image", img5: "image", img6: "image", img7: "image", img8: "image", img9: "image", img10: "image"}
    dzionario={ sample: "testo"}
    asyncio.run(run_pipeline(dzionario))







