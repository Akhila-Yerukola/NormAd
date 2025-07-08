from vllm import LLM, SamplingParams
from typing import List, Union, Tuple
import logging 
import traceback
import pdb
from .base_model import BaseModel
from fastchat.conversation import Conversation, SeparatorStyle
import math
import numpy as np
logger = logging.getLogger(__name__)

COLOR_RED = '\033[91m'
COLOR_GREEN = '\033[92m'
COLOR_BLUE = '\033[94m'
COLOR_NORMAL = '\033[0m'

class LlamaInferencer(BaseModel):
    def __init__(self, model_name_or_path='meta-llama/llama-2-13b-chat-hf', tensor_parallel_size=1, max_tokens=512):
        self.llm = LLM(model_name_or_path, tensor_parallel_size=tensor_parallel_size)
        self.tokenizer = self.llm.get_tokenizer()
        self.max_tokens = max_tokens
    def __call__(self, X: List, max_tokens: Union[int, None]=None, temperature: int=0, return_probs: bool = True):
        return self.forward(X, max_tokens,temperature, return_probs=return_probs)

    def get_probs_for_calibration(self, responses: List) -> Tuple[List[str], List[List[float]]]:
        log_likelihoods = []
        for response in responses:

            text = response.prompt.strip().split("\n")[-1]
            
            if text not in ["Yes", "No", "Neutral", "Unsure"]:
                raise ValueError(f"Invalid last token, check prompts: {text}")
            
            '''
            Modify for each LLM
            '''
            if text in ['Yes', 'No']:
                log_likelihood = list(response.prompt_logprobs[-1].values())[0]
            else:
                # tokenize the text
                tokenized_full = self.tokenizer.encode("Answer:\n"+text, add_special_tokens=False) # apparently, the newline tokenizes differently
                answer_text = self.tokenizer.encode("Answer:\n", add_special_tokens=False)
                tokenized_text = tokenized_full[len(answer_text):][::-1]
                logprobs = []
                tokenized_text_counter = 0
                for prompt_logprobs in response.prompt_logprobs[::-1]:
                    for token_id, logprob in prompt_logprobs.items():
                        if token_id == tokenized_text[tokenized_text_counter]:
                            logprobs.append(logprob)
                            tokenized_text_counter += 1
                        if tokenized_text_counter == len(tokenized_text):
                            break
                    if tokenized_text_counter == len(tokenized_text):
                        break

                assert len(logprobs) == len(tokenized_text)
                log_likelihood = sum(logprobs)
            log_likelihoods.append(log_likelihood)
        
        norm_likelihoods = []
        true_responses = []
        options = ["Yes", "No", "Neutral", "Unsure"]
        for i in range(0, len(log_likelihoods), 4):
            likelihoods = [math.exp(logprob) for logprob in log_likelihoods[i:i+4]]
            norm_likelihood = [prob/sum(likelihoods) for prob in likelihoods]
            # pdb.set_trace()
            norm_likelihoods.append(norm_likelihood)
            index = np.argmax(norm_likelihood)
            gold_response = responses[i+index].outputs[0].text
            true_responses.append(options[index] + gold_response)

        return true_responses, norm_likelihoods

    def forward(self, X: List, max_tokens: Union[int, None]=None, temperature: int=0, return_probs: bool = True) -> List:
        '''
        Get inference outputs from Llama model
        ''' 
        responses = []
        custom_template = Conversation(
            name="llama-2",
            system_template="[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n",
            roles=("[INST]", "[/INST]"),
            messages=[], # don't add any messages
            sep_style=SeparatorStyle.LLAMA2,
            sep=" ",
            sep2="", # don't separate the assistant tag with sep token
        )
        prompts = self.create_instruct_prompts(X, 'llama-2', custom_template=custom_template, assistant_prefix="# Answer:", to_extract_likelihoods=return_probs)
        prompt_logprobs = 1 if return_probs else None
        print(f"Prompt Logprobs: {prompt_logprobs}")
        sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens, prompt_logprobs=prompt_logprobs)
        responses = self.llm.generate(prompts, sampling_params=sampling_params)
        if return_probs:
            responses, probs = self.get_probs_for_calibration(responses)
            return responses, probs
        else:
            responses = [response.outputs[0].text for response in responses]
            return responses, None
    
    
