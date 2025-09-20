import ollama

class LLM:
    def __init__(self, model="llama3"):
        self.model = model

    def chat(self, messages):
        """Sends a conversation history to the Ollama model and returns the response."""
        try:
            response = ollama.chat(model=self.model, messages=messages)
            return response['message']['content']
        except Exception as e:
            print(f"❌ LLM Error: {e}")
            return "Sorry, I had trouble thinking."