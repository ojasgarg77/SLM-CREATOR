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
def find_pdf_files():
    pdf_list=[]
    base_path=Path(__file__).parent/"dataset"/"pdfs"
    if not base_path.is_dir():
        print("""THE DIRECTORY ("pdfs") DOES NOT EXIST.""")
        return pdf_list
    for file_path in sorted(base_path.glob("*.pdf")):
        print(f"PDF {file_path.name} found.")
        pdf_list.append(file_path)
    return pdf_list

def get_english_tokens():
    english_txt_files=[]
    base_path=Path(__file__).parent /"dataset" / "english_dataset"
    if not base_path.is_dir():
        print("""THE DIRECTORY ("english_dataset") DOES NOT EXIST.""")
        return english_txt_files
    for english_files in sorted(base_path.glob("*.txt")):
        print(f"TXT {english_files.name} found.")
        english_txt_files.append(english_files)
    return english_txt_files

def get_arxiv_tokens():
    arxiv_txt_files=[]
    base_path=Path(__file__).parent /"dataset" / "arxiv_texts"
    if not base_path.is_dir():
        print("""THE DIRECTORY ("arxiv_texts") DOES NOT EXIST.""")
        return arxiv_txt_files
    for arxiv_files in sorted(base_path.glob("*.txt")):
        print(f"TXT {arxiv_files.name} found.")
        arxiv_txt_files.append(arxiv_files)
    return arxiv_txt_files

def total_char_count_dic():  
    output_counts={}  

    pdf_files=find_pdf_files()
    for path in pdf_files:
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


    arxiv_files=get_arxiv_tokens()
    for path in arxiv_files:
        count=0
        all_text=path.read_text(encoding="utf-8",errors="ignore")
        for fancy,plain in LIGATURE_DICTIONARY.items():
            all_text=all_text.replace(fancy,plain)
        for line in all_text.splitlines():
            count+=len(line)
        output_counts[path.name]=count


    english_files=get_english_tokens()
    for path in english_files:
        count=0
        all_text=path.read_text(encoding="utf-8",errors="ignore")
        for fancy,plain in LIGATURE_DICTIONARY.items():
            all_text=all_text.replace(fancy,plain)
        for line in all_text.splitlines():
            count+=len(line)
        output_counts[path.name]=count

    return output_counts

def main():
    dataset_files_and_values=total_char_count_dic()
    total_char_count=int(sum(dataset_files_and_values.values()))
    total_dataset_tokens=total_char_count//4
    print(f"TOTAL TOKENS IN THE DATASET IS {total_dataset_tokens}")
    return total_dataset_tokens

if __name__=="__main__":
    total_tokens=main()