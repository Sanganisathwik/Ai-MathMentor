from groq import Groq
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

client = Groq(api_key=GROQ_API_KEY)

def call_groq(prompt: str) -> str:
    """
    Calls the Groq API (Llama 3.3 70B) and returns the response text.
    """
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are a world-class mathematical reasoning engine."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0,
            max_tokens=4096,
            top_p=1,
            stream=False,
            stop=None,
        )

        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"Error calling Groq API: {str(e)}"
