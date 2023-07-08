#%%
import torch
from transformer_lens import HookedTransformer


torch.set_grad_enabled(False)

MODEL_NAME = "gelu-4l"


#%%
model = HookedTransformer.from_pretrained(MODEL_NAME, device="cpu")
model.eval();

#%%
text_prompt = "I went to university at Michigan"
correct_answer = " State"
incorrect_answer = " University"

prompt = model.to_tokens(text_prompt)
correct_token_id = model.to_single_token(correct_answer)
incorrect_token_id = model.to_single_token(incorrect_answer)

#%%
with torch.inference_mode():
    logits, cache = model.run_with_cache(prompt)

#%%
pre_ln = cache["blocks.3.hook_resid_post"]
post_ln = cache["ln_final.hook_normalized"]
scale = cache["ln_final.hook_scale"]

# Sanity check
assert torch.allclose(model.ln_final(pre_ln), post_ln, atol=1e-5)
assert torch.allclose(
    (pre_ln - pre_ln.mean(dim=-1, keepdim=True)) / scale,
    post_ln,
    atol=1e-5,
)

#%%
torch.manual_seed(420)

# a + b = pre_ln
b = torch.randn_like(pre_ln)
a = pre_ln - b
assert torch.allclose(a + b, pre_ln, atol=1e-5)

# Validate decomposing pre_ln into a and b, then layernorming separately
a_normed = (a - a.mean(dim=-1, keepdim=True)) / scale
b_normed = (b - b.mean(dim=-1, keepdim=True)) / scale
assert torch.allclose(a_normed + b_normed, post_ln, atol=1e-5)
