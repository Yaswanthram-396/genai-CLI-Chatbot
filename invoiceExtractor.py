import os
import dotenv
from openai import OpenAI
from decimal import Decimal

import time

import pydantic 
from pydantic import BaseModel,Field
class InvoiceItem(BaseModel):
    title: str
    description: str
    quantity: int = Field(gt=0)
    unitPrice: float = Field(ge=0)
    lineTotal: float = Field(ge=0)

class ChatResponse(BaseModel):
    name:str
    date:str
    items:list[InvoiceItem]
    quantity:int=Field(gt=0)
    unitPrice:float=Field(gt=0)
    total:float=Field(gt=0)


dotenv.load_dotenv()    

OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
GROQ_API_KEY=os.getenv("GROQ_API_KEY")

openai_client = OpenAI(
    base_url="https://api.openai.com/v1",
    api_key=OPENAI_API_KEY
)

groq_client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY
)

history = [
    {
        "role": "system",
        "content": "Extract the invoice details requested by the user. Keep the structured response concise.",
    }
]

def invoiceExtractor(message):
    history.append({"role":"user","content":message})
    response=openai_client.chat.completions.parse(
                model="gpt-4o-mini",
                response_format=ChatResponse,
                max_completion_tokens=2000,
                messages=history)

    msg=response.choices[0].message
    history.append({"role":"assistant","content":msg.content})
    return msg.parsed if msg.parsed else msg.refusal


while True:
    message=input("Your prompt: ")
    if(message.lower()=="exit"):
        print("good bye")
        break
    
    response = invoiceExtractor(message)

    print(response)
