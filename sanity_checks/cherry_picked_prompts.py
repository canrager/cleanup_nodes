#%%
from transformer_lens import HookedTransformer
from transformer_lens.utils import test_prompt

MODEL_NAME = "gelu-4l"


#%%
model = HookedTransformer.from_pretrained(MODEL_NAME, device="cpu")

#%%
examples = [
    {'text': "It's in the shelf, either on the top or the", 'correct': ' bottom'},
    {'text': "I went to university at Michigan", 'correct': ' State'},
    {'text': "class MyClass:\n\tdef", 'correct': ' __'},
]

#%%
for example in examples:
    text = example['text']
    answer = example['correct']
    test_prompt(text, answer, model, prepend_space_to_answer=False)
    print("# ========================================================================= #\n")
