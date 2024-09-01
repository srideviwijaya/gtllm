import triton_python_backend_utils as pb_utils
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TritonPythonModel:
    def initialize(self, args):
        self.tokenizer = AutoTokenizer.from_pretrained("/models/llamav2/1/", device_map=device) # add model repo
        self.model = AutoModelForCausalLM.from_pretrained("/models/llamav2/1/", device_map=device) # add model repo
        self.model.resize_token_embeddings(len(self.tokenizer))

    def get_prompt(self, message: str, 
                   chat_history: list[tuple[str, str]],
                   system_prompt: str) -> str:
        texts = [f'<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n']

        do_strip = False
        for user_input, response in chat_history:
            user_input = user_input.strip() if do_strip else user_input
            do_strip = True
            texts.append(f'{user_input} [/INST] {response.strip()} </s><s>[INST] ')
        message = message.strip() if do_strip else message
        texts.append(f'{message} [/INST]')
        return ''.join(texts)

    def execute(self, requests):
        responses = []
        for request in requests:
            inputs = pb_utils.get_input_tensor_by_name(request, "prompt")
            
            inputs = inputs.as_numpy()

            DEFAULT_SYSTEM_PROMPT = """You are a helpful AI assistant. Keep answers short and concise but still comprehensive."""
            
            prompts = [self.get_prompt(i.decode(), [], DEFAULT_SYSTEM_PROMPT) for i in inputs]
            self.tokenizer.pad_token = "[PAD]"
            self.tokenizer.padding_side = "left"
            inputs = self.tokenizer(prompts, return_tensors='pt', padding=True).to('cuda')

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
                        
            inference_response = pb_utils.InferenceResponse(
            output_tensors=[
                pb_utils.Tensor(
                    "generated_text",
                    np.array([[o.encode() for o in output]]),
                    )
            ]
            )
            responses.append(inference_response)
        
        return responses

    def finalize(self, args):
        self.generator = None