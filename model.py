from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers

import torch
torch.cuda.is_available()
# Output should be True

access_token = "" ## Add your HF credentials
model = "meta-llama/Llama-2-70b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model, token=access_token)

model = AutoModelForCausalLM.from_pretrained(
    model, 
    token=access_token
)

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto",
)