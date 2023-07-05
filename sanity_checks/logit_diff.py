#%%
import torch
from transformer_lens import HookedTransformer
from transformer_lens.utils import test_prompt

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
resid_final = cache["ln_final.hook_normalized"][0, -1]

logit_correct = logits[0, -1, correct_token_id]
logit_incorrect = logits[0, -1, incorrect_token_id]

# Validate manually calculating logits for correct and incorrect
assert torch.allclose(
    resid_final @ model.W_U[:, correct_token_id] + model.b_U[correct_token_id],
    logit_correct,
    atol=1e-5,
)
assert torch.allclose(
    resid_final @ model.W_U[:, incorrect_token_id] + model.b_U[incorrect_token_id],
    logit_incorrect,
    atol=1e-5,
)

#%%
ground_logit_diff = logit_correct - logit_incorrect

# Validate manually calculating logit diff
logit_diff = (
    resid_final @ (model.W_U[:, correct_token_id] - model.W_U[:, incorrect_token_id])
    + model.b_U[correct_token_id]
    - model.b_U[incorrect_token_id]
)

assert torch.allclose(logit_diff, ground_logit_diff, atol=1e-5)

#%%
def get_logit_diff_function(model, correct_token_id, incorrect_token_id):
    logit_diff_direction = model.W_U[:, correct_token_id] - model.W_U[:, incorrect_token_id]

    logit_diff_bias = 0
    if hasattr(model, "b_U"):
        logit_diff_bias = model.b_U[correct_token_id] - model.b_U[incorrect_token_id]

    def calc_logit_diff(resid_final):
        return resid_final @ logit_diff_direction + logit_diff_bias
    
    return calc_logit_diff


# Validate logit diff function
logit_diff_func = get_logit_diff_function(model, correct_token_id, incorrect_token_id)
assert torch.allclose(logit_diff_func(resid_final), ground_logit_diff, atol=1e-5)
