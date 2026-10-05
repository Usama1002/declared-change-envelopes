"""Zero LoRA adapter (B = 0, the PEFT default initialization): identical function to the incumbent but evaluated through
the adapter code path. Its 'changes' measure the numerical floor of an adapter-vs-incumbent comparison.
With a second argument NOISE > 0 the lora_B entries are drawn from N(0, NOISE^2) instead (a tiny random adapter).
Usage: python src/make_null_adapter.py MODEL [NOISE]"""
import sys, torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import fp
m = sys.argv[1]
noise = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0  # std of lora_B entries: a tiny random weight perturbation
model = AutoModelForCausalLM.from_pretrained(fp.MODELS[m], dtype=torch.bfloat16)
cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], task_type="CAUSAL_LM")
pm = get_peft_model(model, cfg)
if noise > 0:
    torch.manual_seed(0)
    for n, p in pm.named_parameters():
        if "lora_B" in n:
            torch.nn.init.normal_(p, std=noise)
pm.save_pretrained(f"{fp.CKPT}/{m}_" + ("null_adapter" if noise == 0 else f"tiny_adapter_{noise:g}"))
print("saved")
