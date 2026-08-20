# /home/m.lami/FlexFL_adapted/llama/__init__.py
from dataclasses import dataclass
from typing import List, Dict, Any
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

@dataclass
class Message:
    role: str
    content: str

@dataclass
class Dialog:
    messages: List[Message]

class Llama:
    def __init__(self, model, tokenizer, device, assistant_role="assistant"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self._assistant_role = assistant_role

    @classmethod
    def build(cls, ckpt_dir: str, tokenizer_path: str = None, **_):
        model_dir = ckpt_dir or os.environ.get("MODEL_DIR")
        tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)

        quant_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            quantization_config=quant_config,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        device = str(next(model.parameters()).device)

        # Gemma uses "model" role; everything else uses "assistant"
        assistant_role = "model" if "gemma" in model_dir.lower() else "assistant"

        max_pos = getattr(model.config, "max_position_embeddings",
                  getattr(model.config, "max_seq_len",
                  getattr(model.config, "seq_length", 131072)))
        model.generation_config.max_length = max_pos
        return cls(model, tok, device, assistant_role)

    def chat_completion(self, dialogs, max_gen_len=512, temperature=0.6, top_p=0.9):
        outs = []
        for d in dialogs:
            if hasattr(d, "messages"):
                msgs = [{"role": m.role, "content": m.content} for m in d.messages]
            elif isinstance(d, list):
                msgs = d
            else:
                raise TypeError(f"Unsupported dialog type: {type(d)}")

            # Normalize roles for this model's chat template
            normalized = []
            for m in msgs:
                role = m["role"].lower()
                if role == "assistant":
                    role = self._assistant_role
                normalized.append({"role": role, "content": m["content"]})

            is_qwen = "qwen" in type(self.tokenizer).__name__.lower()
            tmpl_kwargs = {"enable_thinking": False} if is_qwen else {}
            prompt = self.tokenizer.apply_chat_template(
                normalized, tokenize=False, add_generation_prompt=True, **tmpl_kwargs
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
