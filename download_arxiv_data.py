import arxiv
from pathlib import Path
import re
import requests

def correct_filename(title:str)->str:
    cleaned_name=re.sub(r'[\\/*?"<>|:]',"",title)
    return re.sub(r'\s+',' ',cleaned_name).strip()

def download_pdfs_arxiv(download_dir:Path,query:str="cat:cs.LG AND 'neural networks'",max_results:int=5)->int:
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
        correct_title=correct_filename(paper.title)
        filename=f"{correct_title}.pdf"
        filepath=download_dir/filename

        if filepath.exists():
            if filepath.stat().st_size>1000:
                print(f"PDF ALREADY EXISTS--{paper.title}")
                continue
            else:
                print(f"FOUND CORRUPT FILE. DELETING AND RETRYING--{paper.title}")
                filepath.unlink()

        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ResearchScript/1.0"}

        try:
            response=requests.get(paper.pdf_url,stream=True,headers=headers,timeout=30)
            content_type=response.headers.get("content-type","").lower()
            if response.status_code==200 and "pdf" in content_type:
                with open(filepath,"wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                if filepath.stat().st_size>1000:
                    print(f"SUCCESS--DOWNLOADED {paper.title}")
                    download_count+=1
                else:
                    print(f"DOWNLOAD RESULTED IN OKB FILE {paper.title}")    
                    filepath.unlink()
            else:
                print(f"FAILED TO DOWNLOAD (STATUS CODE--{response.status_code})")
                if filepath.exists():
                    filepath.unlink()
        except Exception as e:
            print(f"ERROR DOWNLOADING PAPER--{e}")
            if filepath.exists() and filepath.stat().st_size==1000:
                filepath.unlink()
    return download_count

if __name__=="__main__":
    target_directory=Path(__file__).parent/"dataset"/"arxiv_pdfs"
    query_string='cat:cs.LG AND "neural networks"'
    download_pdfs_arxiv(target_directory,query=query_string,max_results=5)