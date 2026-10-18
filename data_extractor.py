import pymupdf
import unicodedata
from pathlib import Path
LIGATURE_DICTIONARY={
    "\ufb00":"ff",
    "\ufb01":"fi",
    "\ufb02":"fl",
    "\ufb03":"ffi",
    "\ufb04":"ffl",
    "\ufb05":"st",
    "\ufb06":"st",
    "\u2217":"",
    "\u2020":"",
    "\u2021":""
} 
def find_files():
    pdf_list=[]
    base_path=Path(__file__).parent
    if not (base_path/"dataset").is_dir():
        print("THE DIRECTORY DOES NOT EXIST.")
        return pdf_list
    for file_path in sorted((base_path/"dataset").glob("*.pdf")):
        print(f"PDF {file_path.name} found.")
        pdf_list.append(file_path)
    return pdf_list

files=find_files()  
output_counts={}  
for path in files:
    count=0
    with pymupdf.open(path) as document:
        all_text=""
        for page in document:
            all_text+=page.get_text()

        for fancy,plain in LIGATURE_DICTIONARY.items():
            all_text=all_text.replace(fancy,plain)
        
        for line in all_text.splitlines():
            count+=len(line)
    output_counts[path.name]=count
print(output_counts)
