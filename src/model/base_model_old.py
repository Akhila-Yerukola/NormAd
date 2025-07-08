from fastchat.conversation import Conversation, get_conv_template
from typing import List, Optional
from datasets import Dataset 
import pdb
class BaseModel:
    def __init__(self):
        pass
    
    def __call__(self, X, batch_size: int = None):
        return self.forward(X)

    def forward(self, X, return_probs: bool = False):
        raise NotImplementedError
    
    def append_instruct_prompt(self, model_name: str, text_prompts: List[str], next_responses: List[str], assistant_suffix: Optional[str] = '', custom_template: Optional[Conversation] = None):
        '''
        For use in CoT scenarios
        @param text_prompts: List of original text prompts from USER
        @param model_name: Name of the model
        @param next_response: List of responses from the model
        @param assistant_suffix: add some text to the model's response (like # Answer)
        @param custom_template: provide a custom conversation template in case you want a suffix added
        '''
        prompts = []
        for initial, next_resp in zip(text_prompts, next_responses):
            if custom_template:
                template = custom_template.copy()
            else:
                template = get_conv_template(model_name)
            template.append_message(role="USER", message=initial)
            template.append_message(role="ASSISTANT", message=next_resp+assistant_suffix)
            prompts.append(template.get_prompt())
        return prompts


    def create_instruct_prompts(self, 
                                text_prompts: List[str], 
                                model_name: str, 
                                custom_template: Optional[Conversation] = None,
                                assistant_prefix: Optional[str] = None,
                                to_extract_likelihoods: bool = False) -> Dataset:
        """
        Create prompts for the given dataset
        @param text_prompts: List of text prompts
        @param model_name: Name of the model
        @return: Dataset of prompts
        """
        prompts = []
        for data in text_prompts:
            if to_extract_likelihoods:
                append_tokens = ["Yes", "No", "Neither"]
                for at in append_tokens: 
                    if custom_template:
                        template = custom_template.copy() # reinitialize the template
                    else:
                        template = get_conv_template(model_name)
                    template.append_message(role=custom_template.roles[0], message=data)
                    if assistant_prefix:
                        template.append_message(role=custom_template.roles[1], message=assistant_prefix + f"\n{at}")
                    prompt = template.get_prompt()
                    prompts.append(prompt)
            else: # for openai models
                if custom_template:
                    template = custom_template.copy() # reinitialize the template
                else:
                    template = get_conv_template(model_name)
                template.append_message(role='USER', message=data)
                if assistant_prefix:
                    template.append_message(role='ASSISTANT', message=assistant_prefix)
                prompt = template.get_prompt()
                prompts.append(prompt)
        return prompts
    
    def create_prompts(self, 
                                text_prompts: List[str], 
                                model_name: str,
                                prefix: Optional[str] = None,
                                to_extract_likelihoods: bool = False) -> Dataset:
        """
        Create prompts for the given dataset
        @param text_prompts: List of text prompts
        @param model_name: Name of the model
        @param prefix: Append some prefix to the prompt
        @param to_extract_likelihoods: Append the options [Yes/No/Neutral/Unsure] to the prompt in addition to any prefix strings
        @return: Dataset of prompts
        """
        prompts = []
        for data in text_prompts:
            if to_extract_likelihoods:
                append_tokens = ["Yes", "No", "Neither"]
                for at in append_tokens:
                    prompt = data + "\n" + prefix + '\n' + at
                    prompts.append(prompt)
            else: # for openai models
                prompt = data + "\n" + prefix
                prompts.append(prompt)
        return prompts