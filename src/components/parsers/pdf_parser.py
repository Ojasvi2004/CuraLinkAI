## here we extracting pdf to text

import os

from configurations.config import PDF_INPUT_DIRECTORY,TEXT_OUTPUT_DIRECTORY
import fitz
from tqdm import tqdm
print(fitz.__file__)


os.makedirs(TEXT_OUTPUT_DIRECTORY,exist_ok=True)

def pdf_to_text(filepath):
    doc=fitz.open(filepath)
    parts=[]
    skippedFirst=3
    skippedLast=3
    for pagenumber in range(skippedFirst,len(doc)-skippedLast):
        page=doc[pagenumber]
        text=page.get_text()
        if Is_Useful_Data(text):
            parts.append(text)
        else:
            continue
    return "\n\n".join(parts)


def Is_Useful_Data(text):
    lines = text.splitlines()
    if len(text.strip()) < 200: 
        return False


    all_caps_lines = sum(1 for line in lines if line.strip().isupper())
    if all_caps_lines / max(len(lines), 1) > 0.5:
        return False


    digits_ratio = sum(c.isdigit() for c in text) / max(len(text), 1)
    if digits_ratio > 0.3:  
        return False

    return True
    


def convert_all_pdfs(PDF_INPUT_DIRECTORY):
    files = [f for f in os.listdir(PDF_INPUT_DIRECTORY) if f.lower().endswith(".pdf")]
    if not files:
        print("No PDFs found in", PDF_INPUT_DIRECTORY)
        return
    else:
        for fname in tqdm(files,desc="Converting PDFs to text"):
            currentfilepath=os.path.join(PDF_INPUT_DIRECTORY,fname)
            print(f"Working on {fname} currently")
            file_content=pdf_to_text(filepath=currentfilepath)
            outputFilename= fname.rsplit(".", 1)[0] + ".txt"
            outputPathname=os.path.join(TEXT_OUTPUT_DIRECTORY,outputFilename)
            with open(outputPathname,'w',encoding="utf-8") as file:
                file.write(file_content)
            print(f"Saved {outputFilename} in {TEXT_OUTPUT_DIRECTORY}")


if __name__=="__main__":
    convert_all_pdfs(PDF_INPUT_DIRECTORY=PDF_INPUT_DIRECTORY)