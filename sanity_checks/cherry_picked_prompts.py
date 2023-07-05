#%%
from transformer_lens import HookedTransformer
from transformer_lens.utils import test_prompt

MODEL_NAME = "gelu-4l"


#%%
model = HookedTransformer.from_pretrained(MODEL_NAME, device="cpu")

#%%
examples = [
    {'text': "It's in the shelf, either on the top or the", 'answer': ' bottom'},
    {'text': "I went to university at Michigan", 'answer': ' State'},
    {'text': "class MyClass:\n    def", 'answer': ' __'},
]

#%%
for example in examples:
    text = example['text']
    answer = example['answer']
    test_prompt(text, answer, model, prepend_space_to_answer=False)
    print("# ========================================================================= #\n")
