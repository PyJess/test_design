import pandas as pd
import json
from pathlib import Path
import math

def read_excel(file_excel):    
    df = pd.read_excel(file_excel)
    df.columns = [col.strip() for col in df.columns]

    TITLE_COL = "Title"
    results = []
    current_test = None

    def clean_dict(d):
        cleaned = {}
        for k, v in d.items():
            if v is None:
                continue
            if isinstance(v, float) and math.isnan(v):
                continue
            if isinstance(v, str) and v.strip() == "":
                continue
            cleaned[k] = v
        return cleaned

    for _, row in df.iterrows():

        titolo = row[TITLE_COL]

        if isinstance(titolo, str) and titolo.strip():

            if current_test is not None:
                results.append(current_test)

            current_test = {
                "Titolo": titolo.strip(),
                "contenuto": []
            }

            row_dict = row.to_dict()
            row_dict.pop(TITLE_COL, None)
            row_dict = clean_dict(row_dict)

            if row_dict:
                current_test["contenuto"].append(row_dict)

        else:
            if current_test is not None:
                row_dict = row.to_dict()
                row_dict.pop(TITLE_COL, None)
                row_dict = clean_dict(row_dict)
                if row_dict:
                    current_test["contenuto"].append(row_dict)

    if current_test:
        results.append(current_test)

    json_output = json.dumps(results, indent=4, ensure_ascii=False)
    output_path = Path(__file__).parent / "output_excel.json"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(json_output)

    return results


# def title_content_extraction(json_file):
#     titles = []
#     contents = []
#     for test in json_file:
#         titles.append(test["Titolo"])
#         for step in test["contenuto"]:
#             contents.append(step)
#     return titles, contents

def title_content_extraction(json_file):
    titles = []
    contents = []
    for test in json_file:
        titles.append(test["Titolo"])
        # CORREZIONE: Appende l'intera lista di step/contenuto associata al titolo.
        contents.append(test["contenuto"]) 
    return titles, contents

async def pipeline_excel(file_excel):
    json_file = read_excel(file_excel)
    titles, contents = title_content_extraction(json_file)
    return titles, contents

