import transformers

# Load model directly: https://huggingface.co/gpt2 
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

prompt = "You are playing a game of chess. Respond to the following sequence of moves so that you win the game."

as_white = "You are white, so you move first"

as_black = "You are black. 1: d4"