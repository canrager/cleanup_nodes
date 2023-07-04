# %%
import torch
from transformer_lens import HookedTransformer
from transformer_lens.utils import test_prompt


torch.set_grad_enabled(False)
device = "cpu"

# %%
model = HookedTransformer.from_pretrained("gelu-4l")
model.cfg.use_attn_result = True

# %%
text = "I went to university at Michigan"
tokens = model.tokenizer(text, return_tensors="pt")["input_ids"].to(device)

# %%
_, cache = model.run_with_cache(tokens)

# %%
# Double check manual computing of post-layernorm final residual stream
x = cache["blocks.3.hook_resid_post"]
assert torch.allclose(
    (x - x.mean(dim=-1, keepdim=True)) / cache["ln_final.hook_scale"],
    cache["ln_final.hook_normalized"],
    atol=1e-5,
)
print("Passed check for post-layernorm final residual stream!")
