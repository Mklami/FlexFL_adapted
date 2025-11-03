# /home/m.lami/FlexFL_adapted/llama/__init__.py
from dataclasses import dataclass
from typing import List, Dict, Any
import os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

@dataclass
class Message:
    role: str
    content: str

@dataclass
class Dialog:
    messages: List[Message]

class Llama:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    @classmethod
    def build(cls, ckpt_dir: str, tokenizer_path: str = None, **_):
        model_dir = ckpt_dir or os.environ.get("MODEL_DIR")
        tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype=dtype, low_cpu_mem_usage=True
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        return cls(model, tok, device)

    def chat_completion(self, dialogs, max_gen_len=512, temperature=0.6, top_p=0.9):
        outs = []
        for d in dialogs:
            # Accept either Dialog(...) or list[{"role":..., "content":...}, ...]
            if hasattr(d, "messages"):  # Dialog
                msgs = [{"role": m.role, "content": m.content} for m in d.messages]
            elif isinstance(d, list):   # already list of dicts
                msgs = d
            else:
                raise TypeError(f"Unsupported dialog type: {type(d)}")

            prompt = self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            gen = self.model.generate(
                **inputs,
                max_new_tokens=max_gen_len,
                do_sample=(temperature is not None and temperature > 0),
                temperature=temperature if (temperature is not None) else 0.0,
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id
            )
            gen_ids = gen[0][inputs["input_ids"].shape[1]:]
            text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            outs.append({"generation": {"role": "assistant", "content": text}})
        return outs


