from openai import OpenAI
import os
from tqdm import tqdm
import json

class Models:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.system = """You are a knowledgeable customer support assistant specializing in resolving bank transaction disputes. 
Your role is to guide customers through the process of addressing their concerns effectively, providing concise, empathetic, 
and professional responses. Ensure accuracy, clarity, and adherence to UK banking regulations."""

    def run_chat(self, user, model="gpt-4o-mini", temperature = 0.7):
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": self.system}, {"role": "user", "content": user}],
            temperature=temperature
        )
        message = response.choices[0].message.content
        return message
    
    def run_multiple_chat(self, user, model="gpt-4o-mini", n_responses=2, temperature=1):
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": self.system}, {"role": "user", "content": user}],
            n=n_responses,
            temperature=temperature
        )
        message_1 = response.choices[0].message.content
        message_2 = response.choices[1].message.content
        return message_1, message_2


    def collect_dpo_preferences(self, questions, model, output_file="data/fine_tune_dpo.jsonl"):
        """
        Collect preferences between two model responses for DPO training.
        """
        dpo_examples = []

        for question in tqdm(questions, desc="Collecting preferences"):
            response_1, response_2 = self.run_multiple_chat(
                user=question,
                model=model,
                n_responses=2,
                temperature=0.7
            )
        
            print(f"\nQuestion: {question}")
            print(f"\nResponse 1: {response_1}")
            print(f"Response 2: {response_2}")
        
            # Get user preference
            while True:
                preference = input("\nWhich response do you prefer? (1/2): ").strip()
                if preference in ['1', '2']:
                    break
                print("Please enter either 1 or 2")
        
            # Create DPO example in the correct format
            example = {
                "input": {
                    "messages": [{"role": "user", "content": question}],
                    "tools": [],
                    "parallel_tool_calls": True
                },
                "preferred_output": [
                    {"role": "assistant", "content": response_1 if preference == '1' else response_2}
                ],
                "non_preferred_output": [
                    {"role": "assistant", "content": response_2 if preference == '1' else response_1}
                ]
            }
            dpo_examples.append(example)

        # Save to JSONL file
        with open(output_file, 'w') as f:
            for example in dpo_examples:
                f.write(json.dumps(example) + '\n')

        print(f"\nDPO training file has been saved to {output_file}")
        return dpo_examples
