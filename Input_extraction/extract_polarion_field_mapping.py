import pandas as pd
import os

path=os.path.dirname(__file__)

def extract_field_mapping() -> pd.DataFrame:
    """Extraction of the table with the Type, Label, ID and Field columns """

    print(path)
    df = pd.read_excel(os.path.join(path, 'polarionfieldmapping.xlsx'))
    # Delete the Field column
    df = df.iloc[:, :-1] 
    #print(df)
    return df

# file=extract_field_mapping()
# print(file)
