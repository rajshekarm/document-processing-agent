
import os
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI

from tools.ocr import ocr_read_document


load_dotenv()

# Initialize Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Register OCR tool
tools = [ocr_read_document]

# Create agent
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="""
    You are an intelligent document assistant.

    Your responsibilities:
    1. Extract text from PDF and image documents.
    2. Use ocr_read_document when the user
       requests document text extraction.
    3. Summarize extracted documents.
    4. Answer questions based on document content.
    5. Never invent information missing from documents.
    """
)

# Interactive chatbot
chat_history = []

print("Gemini OCR Agent Ready!")
print("Type 'exit' to quit.\n")

while True:

    user_input = input("You: ").strip()

    if user_input.lower() in ["exit", "quit"]:
        break

    if not user_input:
        continue

    chat_history.append({
        "role": "user",
        "content": user_input
    })

    response = agent.invoke({
        "messages": chat_history
    })

    print("\nAgent:")
    print(response["messages"][-1].content)
    print()

    chat_history = response["messages"]
