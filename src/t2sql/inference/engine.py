"""Local LLM inference for text-to-SQL (design §6).

Single wrapper: load once (4-bit NF4 on the 4 GB GPU), generate with greedy
decoding, extract the SQL statement from free-form model output. The model
is loaded lazily so prompt/split code never pays for it.
"""
from __future__ import annotations

import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

_FENCED = re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL)
_FIRST_SQL = re.compile(r"\b(WITH|SELECT)\b", re.IGNORECASE)


def extract_sql(text: str) -> str:
    """Pull the first SQL statement out of a model response."""
    m = _FENCED.search(text)
    if m:
        text = m.group(1)
    m = _FIRST_SQL.search(text)
    if m:
        text = text[m.start():]
    # one statement: cut at the first statement-terminator followed by more SQL
    stmt = re.split(r";\s*\n", text, maxsplit=1)[0].strip().rstrip(";").strip()
    return stmt


class LLM:
    def __init__(self, model_id: str, *, quantize: bool = True,
                 adapter: str | None = None) -> None:
        self.model_id = model_id
        self.quantize = quantize
        self.adapter = adapter
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        kwargs: dict = {"device_map": "auto"}
        if quantize:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        if adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_new_tokens: int = 256,
        do_sample: bool = False,
        temperature: float = 1.0,
    ) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        gen_kwargs: dict = {"max_new_tokens": max_new_tokens,
                           "pad_token_id": self.tokenizer.eos_token_id}
        if do_sample:
            gen_kwargs.update(do_sample=True, temperature=temperature)
        else:
            gen_kwargs["do_sample"] = False
        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)
        return self.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

    @classmethod
    def cached_model_dir(cls, model_id: str) -> Path | None:
        """Is the snapshot already on disk? (tests avoid a surprise download.)"""
        from huggingface_hub.constants import HF_HUB_CACHE

        hit = next(Path(HF_HUB_CACHE).glob(f"models--{model_id.replace('/', '--')}/snapshots/*"), None)
        return hit