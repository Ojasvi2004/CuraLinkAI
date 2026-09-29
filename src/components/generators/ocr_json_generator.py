from langchain_core.prompts import (PromptTemplate,
                                    ChatPromptTemplate,
                                    ChatMessagePromptTemplate)
from langchain_core.output_parsers import (StrOutputParser)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.runnables import (RunnableParallel,
                                      RunnableMap,
                                      Runnable,
                                      RunnableLambda,
                                      RunnableSequence)
from dotenv import load_dotenv
import sys
load_dotenv()
import os

from openai import OpenAI




from components.parsers.ocr_parser import OCRReports
from components.prompts.ocr_json_prompt import metadata_ask



llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)


def ocr_chain(path):
    ocr_object = OCRReports(path)
    img = ocr_object.load_image()
    print("Image shape:", img.shape) 
    extracted_text = ocr_object.run()
    print("Extracted text:", extracted_text[:200] if extracted_text else "STILL EMPTY")
    return extracted_text
    


def llm_for_ocr(x):
    final_prompt=metadata_ask.invoke({
        "extracted_text":x
    })
    return llm.invoke(final_prompt)


ocr_main_pipeline=RunnableSequence(
    RunnableLambda(ocr_chain),
    RunnableLambda(llm_for_ocr),
    StrOutputParser()
)

if __name__=="__main__":
    result=ocr_main_pipeline.invoke('test\BLR-0425-PA-0040652_LAB MERG_27-04-2025_1239-18_PM@E.pdf_page_7.png')
    print(result)
    
    
    
    

