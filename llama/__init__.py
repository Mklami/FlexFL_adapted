# /home/m.lami/FlexFL_adapted/llama/__init__.py
import os
import re
from dataclasses import dataclass
from typing import List, Dict, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


@dataclass
class Message:
    role: str
    content: str


@dataclass
class Dialog:
    messages: List[Message]


# Matches a leading reasoning block for models that emit one, e.g.
# <think>...</think>, <thought>...</thought>. Non-greedy, DOTALL, tolerant
# of surrounding whitespace. Only strips a block at the very start of output.
_THINK_BLOCK = re.compile(
    r"^\s*<(think|thought)>.*?</\1>\s*", re.DOTALL | re.IGNORECASE
)


def _strip_thinking(text: str) -> str:
    # Remove a leading reasoning block if the template/model produced one
    # despite enable_thinking=False (Gemma 4 occasionally reverts on hard
    # prompts). Also drop any stray unclosed opener at the start.
    cleaned = _THINK_BLOCK.sub("", text)
    cleaned = re.sub(r"^\s*<(think|thought)>\s*", "", cleaned,
                     flags=re.IGNORECASE)
    return cleaned.strip()


class Llama:
    def __init__(self, model, tokenizer, device, assistant_role="assistant",
                 disable_thinking=False):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self._assistant_role = assistant_role
        self._disable_thinking = disable_thinking

    @classmethod
    def build(cls, ckpt_dir: str, tokenizer_path: str = None, **_):
        model_dir = ckpt_dir or os.environ.get("MODEL_DIR")
        tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)

        # Quantization is now opt-in. On the RTX 6000 Pro (96 GB) a 12B in
        # bf16 fits with room to spare and avoids dequant overhead, so the
        # default is full bf16. Set FLEXFL_QUANT=8bit or 4bit to force
        # bitsandbytes on the 4090 (24 GB) if you need it there.
        quant = os.environ.get("FLEXFL_QUANT", "").lower()
        load_kwargs: Dict[str, Any] = dict(
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        if quant == "8bit":
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True)
        elif quant == "4bit":
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        else:
            load_kwargs["torch_dtype"] = torch.bfloat16

        model = AutoModelForCausalLM.from_pretrained(model_dir, **load_kwargs)
        device = str(next(model.parameters()).device)

        name = model_dir.lower()
        is_gemma = "gemma" in name

        # Gemma uses "model" role; everything else uses "assistant".
        assistant_role = "model" if is_gemma else "assistant"

        # Models with a default-on reasoning mode need enable_thinking=False
        # passed to the chat template AND a post-hoc strip as a safety net.
        # Gemma 4 and Qwen3+ both fall in this bucket.
        is_qwen = "qwen" in type(tok).__name__.lower() or "qwen" in name
        disable_thinking = is_gemma or is_qwen

        max_pos = getattr(model.config, "max_position_embeddings",
                  getattr(model.config, "max_seq_len",
                  getattr(model.config, "seq_length", 131072)))
        model.generation_config.max_length = max_pos

        return cls(model, tok, device, assistant_role,
                   disable_thinking=disable_thinking)

    def _apply_template(self, normalized):
        # Try passing enable_thinking; fall back if this template rejects it.
        if self._disable_thinking:
            try:
                return self.tokenizer.apply_chat_template(
                    normalized, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                pass
        return self.tokenizer.apply_chat_template(
            normalized, tokenize=False, add_generation_prompt=True,
        )

    def chat_completion(self, dialogs, max_gen_len=512, temperature=0.2,
                        top_p=0.9):

        outs = []
        for d in dialogs:
            if hasattr(d, "messages"):
                msgs = [{"role": m.role, "content": m.content}
                        for m in d.messages]
            elif isinstance(d, list):
                msgs = d
            else:
                raise TypeError(f"Unsupported dialog type: {type(d)}")

            # Normalize roles for this model's chat template.
            normalized = []
            for m in msgs:
                role = m["role"].lower()
                if role == "assistant":
                    role = self._assistant_role
                normalized.append({"role": role, "content": m["content"]})

            prompt = self._apply_template(normalized)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            do_sample = temperature is not None and temperature > 0
            gen = self.model.generate(
                **inputs,
                max_new_tokens=max_gen_len,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            gen_ids = gen[0][inputs["input_ids"].shape[1]:]
            text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

            if self._disable_thinking:
                text = _strip_thinking(text)

            outs.append({"generation": {"role": "assistant", "content": text}})
        return outs