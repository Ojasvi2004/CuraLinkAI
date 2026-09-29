## dividig into chunks  

import os
import json
from configurations.config import TEXT_OUTPUT_DIRECTORY,CHUNK_OVERLAP,CHUNK_SIZE,CHUNKS_DIRECTORY
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

os.makedirs(CHUNKS_DIRECTORY,exist_ok=True)

MySplitter=RecursiveCharacterTextSplitter(
     chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", " ", ""]
)

def chunk_text_file(filePath):
    with open(filePath,'r',encoding='utf-8') as file:
        text=file.read()
    chunks=MySplitter.split_text(text)
    return chunks


def Chunking_All_File(directory_path):
    files=[f for f in os.listdir(directory_path) if f.lower().endswith(".txt")]
    if not files:
        print("No PDFs found in", directory_path)
        return
    else:
        total=0
        for filename in tqdm(files,desc="Chunking Files"):
            filePath=os.path.join(directory_path,filename)
            print(f"Working on {filename} currently")
            chunks=chunk_text_file(filePath)
            outputFilename= filename.rsplit(".", 1)[0] + "_chunks.json"
            outputPath=os.path.join(CHUNKS_DIRECTORY,outputFilename)
            with open(outputPath,'w',encoding="utf-8") as file:
                json.dump(chunks,file,ensure_ascii=False,indent=2)
            print(f"[OK] {len(chunks)} chunks written to {outputPath}")
            total=total+len(chunks)
        print(f"[FINAL] Total chunks: {total}")

if __name__=="__main__":
    Chunking_All_File(TEXT_OUTPUT_DIRECTORY)
        