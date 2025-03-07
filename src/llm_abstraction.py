import os
import google.generativeai as genai
from abc import ABC, abstractmethod
from dotenv import load_dotenv

class BaseLLM(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass

class GeminiLLM(BaseLLM):
    def __init__(self):
        # Load environment variables from .env file
        load_dotenv()
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not found. Please check your .env file.")
            
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-pro')
    
    def _is_safe(self, response):
        banned_keywords = ["harmful", "biased", "hate speech"]
        return not any(kw in response.lower() for kw in banned_keywords)
    
    def generate(self, prompt: str) -> str:
        response = self.model.generate_content(prompt)
        if self._is_safe(response.text):
            return response.text
        return "Response filtered due to content policy."

# Example usage:
# llm = GeminiLLM()
# print(llm.generate("Hello!"))