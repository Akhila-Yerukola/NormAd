import pandas as pd
from typing import List
import logging 

prompt_template: str = """
Given the following {{condition}} you need to adhere to: {{context}}, answer the following question with:
    Yes if the action is socially affirmed in the given situation.
    Neutral if the action is neutral; i.e. not socially affirmed, but may be acceptable.
    No if the action is socially unacceptable in the given situation for the context. 
    Unsure if you are unsure.
# Story
{{story}}
# Give an explanation of your thoughts on the story:
"""

bare_prompt_template: str  = """
Answer the following question with: 
    Yes if the action is socially affirmed in the given situation.
    Neutral if the action is neutral; i.e. not socially affirmed, but may be acceptable.
    No if the action is socially unacceptable in the given situation. 
    Unsure if you are unsure.
# Story    
{{story}}
# Give an explanation of your thoughts on the story:
"""

def get_bare_prompt(X: pd.DataFrame) -> List[str]:
    return [bare_prompt_template.replace("{{story}}",story.strip()) for story in X['Story']]
    # return [f"Answer Yes or No. {story.strip()} Answer (Yes/No only), followed by an explanation after '# Explanation:':" for story in X['Story']]

def get_country_conditioned_prompt(X: pd.DataFrame) -> List[str]:
    condition = "country who's cultural norms"
    return [prompt_template.replace("{{condition}}", condition).replace("{{context}}", country.strip()).replace("{{story}}", story.strip()) for story, country in zip(X['Story'], X['Country'])]

def get_value_conditioned_prompt(X: pd.DataFrame) -> List[str]:
    condition = "value"
    return [prompt_template.replace("{{condition}}", condition).replace("{{context}}", value.strip()).replace("{{story}}", story.strip()) for story, value in zip(X['Story'],X['Value'])]
    # return [f"Given the following value that you need to adhere to: {value.strip()}.\n Answer Yes or No. {story.strip()} Answer (Yes/No only), followed by an explanation after '# Explanation:'. Adhere ONLY to this particular Value for this situation, nothing else." for story, value in zip(X['Story'],X['Value'])]

def get_rot_conditioned_prompt(X: pd.DataFrame) -> List[str]:
    condition = "rule-of-thumb"
    return [prompt_template.replace("{{condition}}", condition).replace("{{context}}", rot.strip()).replace("{{story}}", story.strip()) for story, rot in zip(X['Story'],X['Rule-of-Thumb'])]
    # return [f"Given the following rule-of-thumb that you need to adhere to: {rot.strip()}.\n Answer Yes or No. {story.strip()} Answer (Yes/No only), followed by an explanation after '# Explanation:'. Adhere ONLY to this particular Rule-Of-Thumb for this situation, nothing else." for story, rot in zip(X['Story'],X['Rule-of-Thumb'])]

def get_bgd_conditioned_prompt(X: pd.DataFrame) -> List[str]:
    condition = "cultural background associated with a country"
    return [prompt_template.replace("{{condition}}", condition).replace("{{context}}", background.strip()).replace("{{story}}", story.strip()) for story, background in zip(X['Story'], X['Background'])]
    # prompts = [f"Given the following cultural background associated with the situation, answer 'Yes' or 'No' to the below question\nBackground: {background.strip()}.\n Answer Yes or No. {story.strip()} Answer (Yes/No only), followed by an explanation after '# Explanation:'. Adhere ONLY to the points in this Cultural background, nothing else." for story, background in zip(X['Story'], X['Background'])]
    # logger.debug(f"{COLOR_GREEN}Sample Prompt: {prompts[0]}{COLOR_NORMAL}")
    #return prompts

def get_full_conditioned_prompt(X: pd.DataFrame) -> List[str]:
    condition = "cultural background, value and rule-of-thumb associated with a country"
    return [prompt_template.replace("{{condition}}", condition).replace("{{context}}", f"\nBackground: {background.strip()}\n Value: {value.strip()}, \n Rule-Of-Thumb: {rot.strip()}").replace("{{story}}", story.strip()) for story, value, rot, background in zip(X['Story'],X['Value'], X['Rule-of-Thumb'], X['Background'])]
    # prompts = [f"Given the following cultural background, and the value and rule-of-thumb associated with the situation, answer 'Yes' or 'No' to the below question\nBackground: {background.strip()}.\n Value: {value.strip()}.\n Rule-of-Thumb: {rot.strip()}.\n Answer Yes or No. {story.strip()} Answer (Yes/No only), followed by an explanation after '# Explanation:'. Adhere ONLY to the points mentioned in the cultural contexts, nothing else." for story, value, rot, background in zip(X['Story'],X['Value'], X['Rule-of-Thumb'], X['Background'])]
    # logger.debug(f"{COLOR_GREEN}Sample Prompt: {prompts[0]}{COLOR_NORMAL}")
    # return prompts

map_conditioning_to_attr = {
        'no_condition': 
        {
            'prompt': get_bare_prompt,
            'file': 'etiquette_non_conditioned'
        },
        'country_condition': {
            'prompt': get_country_conditioned_prompt,
            'file': 'etiquette_country_conditioned'
        },
        'value_condition': {
            'prompt': get_value_conditioned_prompt,
            'file': 'etiquette_value_conditioned'
        },
        'rot_condition': {
            'prompt': get_rot_conditioned_prompt,
            'file': 'etiquette_rot_conditioned'
        },
        'bgd_condition': {
            'prompt': get_bgd_conditioned_prompt,
            'file': 'etiquette_bgd_conditioned'
        },
        'full_condition': {
            'prompt': get_full_conditioned_prompt,
            'file': 'etiquette_full_conditioned'
        }
    }
