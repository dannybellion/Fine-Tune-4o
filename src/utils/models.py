from openai import OpenAI
import os

class Models:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def run_chat(self, system, user, model="gpt-4o-mini"):
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        message = response.choices[0].message.content
        return message
    
    def customer_service_chat(self, user, model="gpt-4o-mini"):
        system = """You are a knowledgeable customer support assistant specializing in resolving bank transaction disputes. 
Your role is to guide customers through the process of addressing their concerns effectively, providing clear, empathetic, 
and professional responses. Ensure accuracy, clarity, and adherence to UK banking regulations."""
        return self.run_chat(system, user, model)
