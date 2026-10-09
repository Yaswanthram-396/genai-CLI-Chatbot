import os
import dotenv
from openai import OpenAI
import time

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
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello! How can I assist you today?"}
]

def chat(message):
    history.append({"role": "user", "content": message})
    for attempt in range(3):

        try:
            print("\nOpen ai Responding")
            response=openai_client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.2,
                messages=history)
            
            return {
        "answer": response.choices[0].message.content,
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens
    }
        
        except Exception as error:
            print("\nOpenAi Failed: ",error)

            if(attempt<2):
                time.sleep(2*attempt)
    print("\nOpenAI failed. Switching to Gemini...")

    try:
        print("\nGroq Responding")
        response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=history
    )
        return {
         "answer": response.choices[0].message.content,
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
    }
    except Exception as error:
        print("\Groq Failed: ",error)
        history.append({"role": "assistant", "content": "Sorry, both AI providers failed."})
        return "Sorry, both AI providers failed."


while True:
    message=input("Your prompt: ")
    if(message.lower()=="exit"):
        print("good bye")
        break
    
    response = chat(message)

    print(response)

