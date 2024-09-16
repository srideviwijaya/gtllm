import triton_python_backend_utils as pb_utils
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import huggingface_hub

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

huggingface_hub.login(token="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") ## Add your HF credentials

class TritonPythonModel:
    def initialize(self, args):
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", device_map=device)
        # self.tokenizer = AutoTokenizer.from_pretrained("/models/llamav2/1/f5db02db724555f92da89c216ac04704f23d4590", device_map=device)
        self.quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        self.model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf",  
                                                          device_map=device,
                                                          quantization_config=self.quantization_config)
        self.model.resize_token_embeddings(len(self.tokenizer))

    def get_prompt(self, message: str, 
                   chat_history: list[tuple[str, str]],
                   system_prompt: str) -> str:
        texts = [f'<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n']
        # The first user input is _not_ stripped
        do_strip = False
        for user_input, response in chat_history:
            user_input = user_input.strip() if do_strip else user_input
            do_strip = True
            texts.append(f'{user_input} [/INST] {response.strip()} </s><s>[INST] ')
        message = message.strip() if do_strip else message
        texts.append(f'{message} [/INST]')
        return ''.join(texts)

    def execute(self, requests):
        """Process batch requests for LLaMA2 inference."""

        # List to accumulate the batched input token tensors
        input_texts = []
        DEFAULT_SYSTEM_PROMPT = """You are a helpful AI assistant. Keep answers short and concise but still comprehensive."""

        # Process each request in the batch
        for request in requests:
            # Get the input tensor by name (e.g., 'input_ids')
            input_tensor = pb_utils.get_input_tensor_by_name(request, 'INPUT_TEXT')

            # Convert the input tensor to a numpy array (assumed to be a string)
            input_data = input_tensor.as_numpy()

            # Decode input bytes to string (necessary for text tokenization)
            for byte_string in input_data:
                if isinstance(byte_string, bytes):
                    input_text = byte_string.decode('utf-8')
                prompt = byte_string
                # prompt = self.get_prompt(input_text, [], DEFAULT_SYSTEM_PROMPT)
                # input_text = byte_string.decode("utf-8")  # Decode each byte string to text
                input_texts.append(str(prompt))

        # Tokenize the batch of input texts for LLaMA2
        self.tokenizer.pad_token = "[PAD]"
        self.tokenizer.padding_side = "left"

        inputs = self.tokenizer(input_texts, return_tensors='pt', padding=True).to(device)

        output_sequences = self.model.generate(
            **inputs,
            do_sample=True,
            max_length=3584,
            temperature=0.01,
            top_p=1,
            top_k=20,
            repetition_penalty=1.1
            )

        output = self.tokenizer.batch_decode(output_sequences, skip_special_tokens=True)

        # Move input tensors to the correct device
        # input_ids = inputs["input_ids"].to(device)
        # attention_mask = inputs["attention_mask"].to(device)

        # Perform inference using the LLaMA2 model
        # with torch.no_grad():
        #     outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Get the logits or perform greedy decoding for generated text
        # Here we will generate responses using greedy decoding
        # generated_tokens = torch.argmax(outputs.logits, dim=-1)

        # Convert the generated tokens back into human-readable text
        # output_texts = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        # List to store the responses for each request
        responses = []

        # Iterate through each request and prepare the output response
        for i, request in enumerate(requests):
            # Create the output tensor from the generated text
            output_tensor = pb_utils.Tensor('OUTPUT_TEXT', np.array(output[i], dtype=np.object_))

            # Create an inference response
            response = pb_utils.InferenceResponse(output_tensors=[output_tensor])

            # Append the response
            responses.append(response)

        return responses

    def finalize(self):
        self.generator = None