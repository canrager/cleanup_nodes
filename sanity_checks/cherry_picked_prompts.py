#%%
from transformer_lens import HookedTransformer
from transformer_lens.utils import test_prompt

IPSUM = "Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum."

MODEL_NAME = "gelu-4l"


#%%
model = HookedTransformer.from_pretrained(MODEL_NAME, device="cpu")

#%%
examples = [
    {
        "text": 4 * IPSUM + " It's in the cupboard, either on the top or the",
        'correct': ' bottom'
    },
    {
        "text": 5 * IPSUM + " I went to university at Michigan",
        'correct': ' State'
    },
    {
        "text": IPSUM + " class MyClass:\n\tdef",
        'correct': ' __'
    },
    {
        "text": 6 * IPSUM + "The church I go to is the Seventh-day Adventist",
        "correct": " Church",
    },
]

#%%
for example in examples:
    text = example['text']
    answer = example['correct']
    test_prompt(text, answer, model, prepend_space_to_answer=False)
    print("# ========================================================================= #\n")
