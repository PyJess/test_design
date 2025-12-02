import pandas as pd
from pathlib import Path

file_excel = Path(__file__).parent/"generated_test_cases3_video_showcase_feedbackAI_requisiti_feedbackAI_requisiti_excel_fixed (4).xlsx"

# Percorso al file CSV/Excel. Usiamo il nome del file che hai caricato.
# Se il tuo file è un .xlsx, devi cambiare 'read_csv' in 'read_excel'

# Nomi delle colonne dedotti dalla struttura del tuo file.
# DEVONO ESSERE ESATTI (case-sensitive) come nella riga di intestazione del tuo file.
COLONNA_ID_TC = 'Test Case ID' # Colonna 1: Verify SGP TC-XXX
COLONNE_STEP_DATA = [
    'Step N°', 
    'Step', 
    'Expected Result', 
    'Platform', 
    'Type'
    # Aggiungi qui tutte le altre colonne che compongono i dettagli dello step
]
# ----------------------


def estrai_test_case_strutturati(file_path: Path, colonna_tc: str, colonne_step: list) -> list:
    """Legge il file e raggruppa gli step sotto i rispettivi Test Case usando ffill."""
    try:
        # 1. Carica i dati. Assumiamo che la prima riga sia l'intestazione
        # Se stai leggendo un CSV, usa read_csv; se un XLSX, usa read_excel.
        if file_path.suffix.lower() == '.csv':
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        
        # Pulizia iniziale: rimuovi righe completamente vuote che potrebbero interferire
        df.dropna(how='all', inplace=True)

        # 2. Riempimento dei valori mancanti (Forward Fill)
        # Riempie le celle vuote nella colonna ID TC con l'ultimo ID valido.
        df[colonna_tc] = df[colonna_tc].ffill()
        
        # Filtra i dati: a volte ci sono righe di intestazione ripetute o righe inutili
        # che ffill() può riempire. Assicuriamoci che ci siano almeno dati di step.
        df.dropna(subset=[colonne_step[0]], inplace=True)

        # 3. Raggruppa i dati per l'ID del Test Case
        test_cases_raggruppati = []
        
        for tc_id, gruppo in df.groupby(colonna_tc):
            steps_list = []
            
            # Itera sulle righe (gli step) del gruppo
            for _, riga_step in gruppo.iterrows():
                
                step_data = {}
                for colonna in colonne_step:
                    # Assicurati che l'informazione sia convertita in stringa per sicurezza (o altro tipo necessario)
                    step_data[colonna] = str(riga_step.get(colonna, ''))
                    
                steps_list.append(step_data)

            # Crea il dizionario finale per il Test Case
            test_case_completo = {
                "Test Case ID": tc_id,
                "Steps": steps_list
            }
            
            test_cases_raggruppati.append(test_case_completo)
            
        return test_cases_raggruppati

    except FileNotFoundError:
        print(f"❌ Errore: File non trovato all'indirizzo {file_path}")
        return []
    except Exception as e:
        print(f"❌ Si è verificato un errore durante la lettura del file: {e}")
        return []

# Esegui la funzione
risultato = estrai_test_case_strutturati(
    file_path=file_excel,
    colonna_tc=COLONNA_ID_TC,
    colonne_step=COLONNE_STEP_DATA
)

# Stampa il risultato in formato JSON per visualizzare la struttura
import json
print("\n--- RISULTATO STRUTTURATO ---\n")
print(json.dumps(risultato, indent=4, ensure_ascii=False))