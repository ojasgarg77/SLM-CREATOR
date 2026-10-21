from datasets import load_dataset
from pathlib import Path
from dotenv import load_dotenv

env_path=Path(__file__).parent/".env"
load_dotenv(dotenv_path=env_path)

def stream_and_save_english_file(target_file:Path,max_sentences:int=500_000)->int:

    if target_file.exists():
        print(f"FILE ALREADY EXISTS AT {target_file}. SKIPPING DOWNLOAD \n")
        return len(target_file.read_text(encoding="utf-8"))

    target_file.parent.mkdir(parents=True,exist_ok=True)

    input("INTERNET CONNECTION IS REQUIRED FOR THIS STEP\n PRESS ENTER TO CONTINUE--")

    dataset_stream=load_dataset("kd13/bookcorpus-clean", split="train", streaming=True)

    with open (target_file,"w",encoding="utf-8") as f:
        sentence_count=0
        total_characters=0

        for row in dataset_stream:
            text_line=row["text"].strip()
            f.write(text_line+"\n")
            total_characters+=len(text_line)
            sentence_count+=1

            if sentence_count % 100_000==0:
                print(f"\nPULLED {sentence_count} SENTENCES.\n")

            if sentence_count>=500_000:
                break
    return total_characters

if __name__=="__main__":
    target_directory=Path(__file__).parent/"dataset"/"english_dataset"
    target_file=target_directory/"english_dataset.txt"
    total_characters=stream_and_save_english_file(target_file)
    print(f"DOWNLOADED {total_characters} TOTAL CHARACTERS.")