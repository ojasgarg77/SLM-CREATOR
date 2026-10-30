import arxiv
from pathlib import Path
import re
import requests
from bs4 import BeautifulSoup
import time
from bs4 import NavigableString

def clean_paper_text(text:str)->str:
    text=text.replace("\u200b","").replace("\u200c","").replace("\u200d","").replace("\u00ad","")
    text=re.sub(r'\s+([\)\]}])',r'\1',text)
    text=re.sub(r"\s+([,\.\?!;:])",r"\1",text)
    text=re.sub(r"\n{3,}","\n\n",text)
    text=re.sub(r'(\w+)-\s*\n\s*(\w+)',r'\1\2', text)
    return text.strip()

def clean_math_output(soup):

    for hidden_span in soup.find_all(class_=re.compile("ltx_visually_hidden|ltx_role_presentation",re.IGNORECASE)):
        hidden_span.decompose()

    math_tags = list(soup.find_all(["math","span","div"],class_=re.compile(r"ltx_math|ltx_Math", re.IGNORECASE)))
    math_tags.sort(key=lambda x:len(list(x.parents)), reverse=True)

    for math_tag in math_tags:

        if not math_tag.parent:
            continue

        latex_annotation=math_tag.find(
            "annotation",{"encoding":"application/x-tex"}
        )

        if latex_annotation:
            clean_latex=f"${latex_annotation.get_text(strip=True)}$"

            if not clean_latex.startswith("$"):
                clean_latex=f"${clean_latex}$"

            math_tag.replace_with(clean_latex)

        else:
            fallback_text = math_tag.get_text(separator="",strip=True)
            if fallback_text:
                math_tag.replace_with(f"${fallback_text}$")
            else:
                math_tag.decompose()
    return soup

def lowercase_txt(text:str)->str:
    return text.lower()

def download_txt_arxiv(download_dir:Path,query:str="cat:cs.LG AND 'neural networks'",max_results:int=3)->int:
    download_dir.mkdir(parents=True,exist_ok=True)
    client=arxiv.Client()
    search_query=arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )

    results=list(client.results(search_query))
    if len(results)==0:
        print("NO RESULTS MATCHING THE QUERY.")
        return 0
    download_count=0

    for i,paper in enumerate(results):
        clean_arxiv_id=paper.entry_id.split("/abs/")[-1].rsplit("v", 1)[0]
        filename = f"{clean_arxiv_id}.txt"
        filepath=download_dir/filename

        if filepath.exists():
            if filepath.stat().st_size>1:
                print(f"FILE ALREADY EXISTS--{paper.title}")
                continue
            else:
                print(f"FOUND CORRUPT FILE. DELETING AND RETRYING--{paper.title}")
                filepath.unlink()

        ar5iv_url="https://ar5iv.labs.arxiv.org/html/"+clean_arxiv_id

        try:
            session=requests.Session()
            session.headers.update({"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) ResearchScript/1.0"})
            response=session.get(ar5iv_url,stream=True,timeout=30)

            if response.status_code==200:
                soup=BeautifulSoup(response.text,"html.parser")
                document_container=soup.find(class_="ltx_document")
                if document_container:

                    for cite_node in document_container.find_all(class_="ltx_cite"):
                        cite_node.replace_with(" [cite] ")

                    for tag_node in document_container.find_all(class_="ltx_tag"):
                        parent=tag_node.parent
                        if parent and parent.has_attr("class"):
                            if any("eqno" in cls for cls in parent["class"]):
                                tag_node.decompose()

                    for junk in document_container.find_all(
                        class_=[
                            "ltx_author", "ltx_authors", "ltx_role_affiliation", "ltx_email", 
                            "ltx_orcid", "ltx_abstract", "ltx_classification", "ltx_bibliography", 
                            "ltx_acknowledgement", "ltx_acknowledgments", "ltx_cite", 
                            "ltx_running_head", "ltx_pagination", "ltx_page_number", 
                            "ltx_history", "ltx_copyright", "ltx_note", "ltx_notes", "ltx_footnote",
                            "ltx_float"
                        ]
                    ):
                        junk.decompose()

                    for table_node in document_container.find_all(class_=["ltx_table","ltx_tabular"]):
                        does_equation_exist=False
                        for descendant in table_node.find_all(class_=True):

                            if any("eqn_row" in cls for cls in descendant["class"]):
                                does_equation_exist=True
                                break

                        if not does_equation_exist:
                            table_node.decompose()
                    
                    process_prose_nodes(document_container)
                    document_container=clean_math_output(document_container)

                    target_classes=[
                    "ltx_p",
                    "ltx_eqn_block",
                    "ltx_equation",
                    "ltx_equationgroup",
                    "ltx_eqn_row",
                    "ltx_displaymath",
                    "ltx_title_section",
                    "ltx_title_subsection",
                    "ltx_title_subsubsection",
                        ]

                    content_elements = document_container.find_all(
                        class_=target_classes
                    )

                    content_set={id(el) for el in content_elements}
                    filtered_elements=[]
                    for el in content_elements:
                        if not any(id(parent) in content_set for parent in el.parents):
                            filtered_elements.append(el)

                    text_blocks=[]
                    last_text=(
                        None
                    )
                    for el in filtered_elements:
                        text=el.get_text(separator=" ", strip=True)
                            
                        if text and text!=last_text:
                            text_blocks.append(text)
                            last_text=text

                    full_paper="\n\n".join(text_blocks)
                else:
                    full_paper=""

                full_paper=clean_paper_text(full_paper)

                if len(full_paper.strip())<=100:
                    print("WARNING--BODY EXTRACTION HAS FAILED.")
                    time.sleep(2)
                    continue


                file_content=(
                    f"title--{paper.title.lower}\n"
                    f"arxiv id--{clean_arxiv_id}\n"
                    f"abstract--{paper.summary.lower}\n"
                    f"body--{full_paper}\n\n")


                with open(filepath,"w",encoding="utf-8") as f:
                    f.write(file_content)

                if filepath.stat().st_size>1:
                    print(f"SUCCESS--DOWNLOADED {paper.title}")
                    download_count+=1
                else:
                    print(f"DOWNLOAD RESULTED IN OKB FILE {paper.title}")    
                    filepath.unlink()
            else:
                print(f"FAILED TO DOWNLOAD (STATUS CODE--{response.status_code})")
                if filepath.exists():
                    filepath.unlink()

            time.sleep(2)

        except Exception as e:
            print(f"ERROR DOWNLOADING PAPER--{e}")
            if filepath.exists() and filepath.stat().st_size<=1:
                filepath.unlink()
    return download_count

def process_prose_nodes(container_nodes):

    LIGATURE_AND_UNICODE_DICTIONARY={
    "\ufb00":"ff",
    "\ufb01":"fi",
    "\ufb02":"fl",
    "\ufb03":"ffi",
    "\ufb04":"ffl",
    "\ufb05":"st",
    "\ufb06":"st",
    "\u2217":"",
    "\u2020":"",
    "\u2021":"",
    "\xa0":" ",
    "\u2009":" ",
    "\u2004":" ",
    } 

    for node in container_nodes.find_all(string=True):

        if not isinstance(node,NavigableString):
            continue
        if node.parent and (node.parent.name in ["math", "annotation"] or "ltx_math" in "".join(node.parent.get("class",[]))):
            continue

        is_math_node=False

        for parent in node.parents:
            if parent.name in ["math", "annotation"]:
                is_math_node=True
                break

            classes=parent.get("class",[])
            if classes and any(re.search(r"ltx_math",str(cls),re.IGNORECASE) for cls in classes):
                is_math_node=True
                break
             
        if is_math_node:
            continue     

        orignal_text=node.string
        if orignal_text:
            cleaned_text=orignal_text.lower()
            for unicode_char,replacements in LIGATURE_AND_UNICODE_DICTIONARY.items():
                cleaned_text=cleaned_text.replace(unicode_char,replacements)
            node.replace_with(cleaned_text)

if __name__=="__main__":
    target_directory=Path(__file__).parent/"dataset"/"arxiv_texts"
    query_string='(cat:cs.CL OR cat:cs.LG) AND ("language model" OR "transformer" OR "sequence modeling" OR "autoregressive" OR "text generation")'
    input("INTERNET CONNECTION IS REQUIRED FOR THIS STEP\n PRESS ENTER TO CONTINUE--")
    while True:
        user_input=input("\nHOW MANY PAPER DO YOU WANT TO DOWNLOAD?\n")
        try:
            num_papers=int(user_input)
            if num_papers>0:
                break
            else:
                print("INVALID INPUT.")
        except ValueError:
            print("INVALID INPUT")
    total_download=download_txt_arxiv(target_directory,query=query_string,max_results=num_papers)   