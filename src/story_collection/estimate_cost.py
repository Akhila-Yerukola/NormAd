import tiktoken
import pandas as pd
import ast
enc = tiktoken.encoding_for_model('gpt-4-turbo')
def num_tokens_from_string(string: str, encoding) -> int:
    """Returns the number of tokens in a text string."""
    num_tokens = len(encoding.encode(string))
    return num_tokens
STORY_PATH = '../../output/Etiquette_gpt4_3opt.csv'
df = pd.read_csv(STORY_PATH)
prompts = df['Prompt'].tolist()
import json
print(prompts[0])
prompts = [ast.literal_eval(p) for p in prompts]
prompts = [" ".join(k['content'] for k in p)  for p in prompts ]
ntokens = [num_tokens_from_string(p, enc) for p in prompts]
df['num_tokens'] = ntokens
print(df['num_tokens'].describe())

# from src.story_collection.utils import PromptBuilder
