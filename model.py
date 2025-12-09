import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class HFModel:
    def __init__(self, model_name="speakleash/Bielik-11B-v2.6-Instruct"):
        self.device = "auto"

        self.model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)#, torch_dtype=torch.bfloat16)

    def generate(self, messages):
        input_ids = self.tokenizer.apply_chat_template(messages, return_tensors="pt")

        model_inputs = input_ids.to(self.device)
        self.model.to(self.device)

        generated_ids = self.model.generate(model_inputs, max_new_tokens=20, do_sample=True)
        decoded = self.tokenizer.batch_decode(generated_ids)
        print(decoded[0])