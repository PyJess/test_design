import docx2pdf
from pdf2image import convert_from_path
import os

def docx_to_image(docx_path, output_folder=None, dpi=200):
    """
    Converte un file DOCX in immagini (una per pagina)
    
    Args:
        docx_path: percorso del file DOCX
        output_folder: cartella di output (default: stessa del DOCX)
        dpi: risoluzione delle immagini (default: 200)
    
    Returns:
        lista dei percorsi delle immagini create
    """
    
    # Imposta cartella di output
    if output_folder is None:
        output_folder = os.path.dirname(docx_path)
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Nome base del file
    base_name = os.path.splitext(os.path.basename(docx_path))[0]
    
    # Step 1: Converti DOCX in PDF temporaneo
    pdf_path = os.path.join(output_folder, f"{base_name}_temp.pdf")
    print(f"Conversione {docx_path} in PDF...")
    docx2pdf.convert(docx_path, pdf_path)
    
    # Step 2: Converti PDF in immagini
    print(f"Conversione PDF in immagini...")
    images = convert_from_path(pdf_path, dpi=dpi)
    
    # Step 3: Salva le immagini
    image_paths = []
    for i, img in enumerate(images, 1):
        img_path = os.path.join(output_folder, f"{base_name}_pagina_{i}.png")
        img.save(img_path, 'PNG')
        image_paths.append(img_path)
        print(f"Salvata: {img_path}")
    
    # Rimuovi PDF temporaneo
    os.remove(pdf_path)
    
    print(f"\nConversione completata! Create {len(image_paths)} immagini.")
    return image_paths



from docx import Document
from docx.oxml import parse_xml
import os

def estrai_immagini_da_docx(docx_path, output_folder=None):
    """
    Estrae tutte le immagini da un file DOCX
    
    Args:
        docx_path: percorso del file DOCX
        output_folder: cartella di output (default: docx_images)
    
    Returns:
        lista dei percorsi delle immagini estratte
    """
    
    # Imposta cartella di output
    if output_folder is None:
        output_folder = "docx_images"
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Apri il documento
    doc = Document(docx_path)
    
    # Nome base del file
    base_name = os.path.splitext(os.path.basename(docx_path))[0]
    
    image_paths = []
    image_count = 0
    
    # Estrai le immagini dalle relazioni del documento
    for rel in doc.part.rels.values():
        if "image" in rel.target_ref:
            image_count += 1
            
            # Ottieni i dati dell'immagine
            image_data = rel.target_part.blob
            
            # Determina l'estensione del file
            content_type = rel.target_part.content_type
            ext = content_type.split('/')[-1]
            if ext == 'jpeg':
                ext = 'jpg'
            
            # Salva l'immagine
            img_filename = f"{base_name}_immagine_{image_count}.{ext}"
            img_path = os.path.join(output_folder, img_filename)
            
            with open(img_path, 'wb') as img_file:
                img_file.write(image_data)
            
            image_paths.append(img_path)
            print(f"Estratta: {img_filename}")
    
    if image_count == 0:
        print("Nessuna immagine trovata nel documento.")
    else:
        print(f"\nTotale immagini estratte: {image_count}")
    
    return image_paths


# Esempio di utilizzo
# if __name__ == "__main__":
#     # Modifica questo percorso con il tuo file DOCX
#     docx_file =r"C:\Users\x.hita\Downloads\Doc2.docx"
    
#     # Estrai le immagini
#     images = estrai_immagini_da_docx(
#         docx_path=docx_file,
#         output_folder="outputs"
#     )
    
#     print(f"\nImmagini salvate in:")
#     for img in images:
#         print(f"  - {img}")


def estrai_immagini_da_pdf(pdf_path, output_folder="image", dpi=200):

    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    images = convert_from_path(pdf_path, dpi=dpi)
    
    # Step 3: Salva le immagini
    image_paths = []
    for i, img in enumerate(images, 1):
        img_path = os.path.join(output_folder, f"{base_name}_paginacreata_{i}.png")
        img.save(img_path, 'PNG')
        image_paths.append(img_path)
        print(f"Salvata: {img_path}")
    
    # Rimuovi PDF temporaneo
    os.remove(pdf_path)
    
    print(f"\nConversione completata! Create {len(image_paths)} immagini.")
    return image_paths

# if __name__ == "__main__":
#     # Modifica questo percorso con il tuo file DOCX
#     docx_file = r"C:\Users\x.hita\Downloads\Doc2.docx"
    
#     # Converti il documento
#     images = docx_to_image(
#         docx_path=docx_file,
#         output_folder="image",  # cartella di output
#         dpi=300  # qualità alta
#     )
    
#     print(f"\nImmagini create:")
#     for img in images:
#         print(f"  - {img}")



if __name__ == "__main__":
    # Modifica questo percorso con il tuo file DOCX
    pdf_file= r"C:\Users\x.hita\Downloads\SVT.pdf"
    
    # Estrai le immagini
    images = estrai_immagini_da_pdf(
        pdf_path=pdf_file,
        output_folder="image"
    )
    
    print(f"\nImmagini salvate in:")
    for img in images:
        print(f"  - {img}")

