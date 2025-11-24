import os
import json
import fitz  
from docx import Document
from dotenv import load_dotenv
from pathlib import Path
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate
from openai import OpenAI

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
    
file_json = load_json(Path(__file__).parent/"regolamento_structured.json")

# Direct access - sezioni is already a list
sezioni = file_json["sezioni"]

# Iterate and find Articolo 6
for sezione in sezioni:
    titolo = sezione.get("titolo", "")
    contenuto = sezione.get("contenuto", "")
    if "articolo 6" in titolo.lower():
        print(f"\n📘 {titolo}\n{'-' * 80}\n{contenuto}\n{'=' * 80}")
        break
else:
    print("⚠️ Articolo 6 non trovato nel documento.")