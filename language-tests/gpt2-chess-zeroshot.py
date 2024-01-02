import transformers

# Load model directly: https://huggingface.co/gpt2 
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")


as_white = "You are playing a game of chess. You are white, so you move first. Make your move. Your objective is to win the game."


as_black = """You are playing a game of chess. You are black. Respond to white's move. Your goal is to win the game."""


isGameFinished=False
prompt = as_white
isWhite=True

while(not isGameFinished):
    inputs = tokenizer.encode(prompt, return_tensors='pt')

    outputs = model.generate(inputs, max_length=50, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(response)
    if(isWhite):
        isWhite=False
        prompt = as_black + response
    else:
        isWhite = True
        prompt = as_white + response