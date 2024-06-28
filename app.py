import ctransformers
from ctransformers import AutoModelForCausalLM


def load_model(model_path):
    return AutoModelForCausalLM.from_pretrained(
        model_path,
        model_type="mistral",
        gpu_layers=0  # Set to a higher number if you have a GPU
    )

def generate_response(model, prompt, max_new_tokens=100):
    return model(prompt, max_new_tokens=max_new_tokens, temperature=0.7)

def main():
    model_path = "path/to/mistral-7b-instruct-v0.2.Q4_K_S.gguf"
    model = load_model(model_path)
    
    print("Chat with Mistral 7B Instruct v0.2 (Type 'exit' to quit)")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        
        prompt = f"Human: {user_input}\n\nAssistant: "
        response = generate_response(model, prompt)
        print("Assistant:", response)

if __name__ == "__main__":
    main()