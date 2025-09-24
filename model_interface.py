from llama_cpp import Llama

class LlamaSOPAssistant:
    def __init__(self, model_path):
        self.llm = Llama(model_path=model_path, n_ctx=2048)

    def query(self, prompt):
        response = self.llm(
            prompt,
            max_tokens=512,
            stop=["\nUser:", "\nAssistant:"],
            echo=False
        )
        return response["choices"][0]["text"].strip()
