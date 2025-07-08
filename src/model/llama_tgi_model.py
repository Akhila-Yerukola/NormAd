from vllm import LLM, SamplingParams
from typing import List, Union, Tuple
import logging 
import traceback
import pdb
from .base_model import BaseModel
from fastchat.conversation import Conversation, SeparatorStyle
import math
import numpy as np
from text_generation import AsyncClient
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
import asyncio
from transformers import AutoTokenizer
import pdb
import aiolimiter
logger = logging.getLogger(__name__)

COLOR_RED = '\033[91m'
COLOR_GREEN = '\033[92m'
COLOR_BLUE = '\033[94m'
COLOR_NORMAL = '\033[0m'


NUM_OPTIONS = 3

class LlamaTGIInferencer(BaseModel):
    def __init__(self, model_url, 
                 model_name_or_path='meta-llama/llama-2-13b-chat-hf', 
                 tensor_parallel_size=1,
                 batch_size=32):
        self.batch_size = batch_size
        self.client = AsyncClient(model_url)
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-chat-hf")
        self.chat = True if 'chat' in model_name_or_path else False
        self.cot = False
        self.to_extract_likelihoods = False

    def __call__(self, X: List, max_tokens: int, batch_size:int,  temperature: int=0, cot: bool=False, to_extract_likelihoods: bool=False):
        return self.forward(X, max_tokens, batch_size, temperature, cot=cot, to_extract_likelihoods=to_extract_likelihoods)

    # Code borrowed from Akhila Yerukola, Thanks!
    def __sum_logprob(self, token_list, return_ppl=False):
        # # if chat, remove last 4 tokens 
        # token_list.reverse() #start from end
        total = 0; l = 0
        # remove the 1 token
        token_list = token_list[1:]
        for token in token_list:
            if token.logprob is not None:
                total+= token.logprob
                l+=1
        if return_ppl:
            return np.exp(-total / l)
        else:
            return -total/l
        
    def __get_probabilities(self, responses):
        prefill = [response.details.prefill for response in responses]
        log_likelihoods = []
        for p_tokens in prefill:
            text = p_tokens[-1].text
            '''
            Modify for each LLM
            '''
            if text in ['Yes', 'No']:
                log_likelihood = p_tokens[-1].logprob
            else:
                # tokenize the text
                # start from the reverse and take in three tokens
                tokenized_neither = self.tokenizer.tokenize("Answer\nNeutral", add_special_tokens=False) # apparently, the newline tokenizes differently
                # tokenized_unsure = self.tokenizer.tokenize("Answer\nUnsure", add_special_tokens=False)
                answer_text = self.tokenizer.tokenize("Answer\n", add_special_tokens=False)
                tokenized_neither = tokenized_neither[len(answer_text):]
                # tokenized_unsure = tokenized_unsure[len(answer_text):]
                # compare with the last few tokens of response
                def compare_tokenized_text(tokenized_text, response_suffix):
                    for i in range(len(tokenized_text)):
                        if tokenized_text[i] != response_suffix[i]:
                            return False
                    return True
                if compare_tokenized_text(tokenized_neither, [resp.text for resp in p_tokens[-len(tokenized_neither):]]):
                    logprobs = [resp.logprob for resp in p_tokens[-len(tokenized_neither):]]
                # elif compare_tokenized_text(tokenized_unsure, [resp.text for resp in p_tokens[-len(tokenized_unsure):]]):
                #     logprobs = [resp.logprob for resp in p_tokens[-len(tokenized_unsure):]]
                else:
                    raise ValueError(f"Invalid last token, check prompts: {text}")
                
                log_likelihood = sum(logprobs)
            log_likelihoods.append(log_likelihood)
        
        norm_likelihoods = []
        true_responses = []
        options = ["Yes", "No", "Neutral"] # "Unsure"]
        for i in range(0, len(log_likelihoods), NUM_OPTIONS):
            likelihoods = [np.exp(logprob) for logprob in log_likelihoods[i:i+3]]
            norm_likelihood = likelihoods #[prob/sum(likelihoods) for prob in likelihoods]
            # pdb.set_trace()
            norm_likelihoods.append(norm_likelihood)
            index = np.argmax(norm_likelihood)
            gold_response = responses[i+index].generated_text
            true_responses.append(options[index] + gold_response)

        return true_responses, norm_likelihoods
    
    async def _generate(self, prompt, limiter, max_tokens, temperature):
        async with limiter:
            for _ in range(10):
                try:
                    return await self.client.generate(
                        prompt=prompt,
                        max_new_tokens=max_tokens,
                        temperature=temperature+1e-3,
                        decoder_input_details=True
                    )
                except Exception as e:
                    print("Exception!!!!!!")
                    logging.warning(e)
                    await asyncio.sleep(20)
                # await asyncio.sleep(20)
        return None

    async def __batch_generate_async(self, samples, max_tokens, temperature):
        return await asyncio.gather(
                    *[
                        self.client.generate(
                            sample,
                            max_new_tokens=max_tokens,
                            temperature=temperature+1e-3,
                            decoder_input_details=True
                        )
                        for sample in samples
                    ]
                )

    def get_predictions(self, prompts, max_tokens, batch_size, temperature=0, to_extract_likelihoods=False) -> Tuple[List[str], List[List[float]]]:

        def chunk(seq_1, seq_2, size=128):
            item =  ((seq_1[pos:pos + size] for pos in range(0, len(seq_1), size)), \
                (seq_2[pos:pos + size] for pos in range(0, len(seq_2), size)))
            return item

        assert batch_size % NUM_OPTIONS == 0, f"Batch size must be a multiple of {NUM_OPTIONS}"

        dialog_ppl = []
        dialog_responses = []
        for pos in tqdm(range(0, len(prompts), batch_size)):
            batch_prompts = prompts[pos:pos + batch_size]
            
            responses = [
                response
                for response in asyncio.run(self.__batch_generate_async(batch_prompts, temperature=temperature, max_tokens=max_tokens))
            ]

            if not self.cot and to_extract_likelihoods:
                true_responses, conditional_logprobs = self.__get_probabilities(responses)
                dialog_ppl.extend(conditional_logprobs)
            else:
                true_responses = [response.generated_text for response in responses]
            
           
            dialog_responses.extend(true_responses)
        
        return dialog_responses, dialog_ppl

    def forward(self, X: List, max_tokens: int, batch_size: int, temperature: int, cot: bool = False, to_extract_likelihoods: bool=False) -> List:
        '''
        Get inference outputs from Llama model
        ''' 
        self.cot = cot
        responses = []
        custom_template = Conversation(
            name="llama-2",
            system_template="[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n",
            roles=("[INST]", "[/INST]\n"),
            messages=[], # don't add any messages
            sep_style=SeparatorStyle.LLAMA2,
            sep=" ",
            sep2="", # don't separate the assistant tag with sep token
        )
        self.to_extract_likelihoods = to_extract_likelihoods
        if self.cot:
            if self.chat:
                prompts = self.create_instruct_prompts(X, 'llama-2', custom_template=custom_template, assistant_prefix="# Explanation", to_extract_likelihoods=False)
            else:
                prompts = self.create_prompts(X, 'llama-2',prefix='', to_extract_likelihoods=False)
    
            cond_responses_expl = self.get_predictions(prompts, batch_size=batch_size, temperature=temperature, max_tokens=max_tokens)
            print("Received Explanations")
            cond_responses_expl = [c.generated_text for c in cond_responses_expl]

            if not self.chat:
                X = [x+cond_resp_expl for x, cond_resp_expl in zip(X, cond_responses_expl)]
                prompts = self.create_prompts(X, 'llama-2', prefix='', to_extract_likelihoods=to_extract_likelihoods)
            else:
                prompts = self.append_instruct_prompt(model_name='llama-2', text_prompts=X, next_responses = cond_responses_expl, assistant_suffix='# Answer (Yes, Neutral, No)\n', custom_template=custom_template)

            self.cot = False
        
        if self.chat:
            prompts = self.create_instruct_prompts(X, 'llama-2', custom_template=custom_template, assistant_prefix="# Answer (Yes, Neutral, No)\n", to_extract_likelihoods=to_extract_likelihoods)
        else:
            prompts = self.create_prompts(X, 'llama-2', prefix ='# Answer (Yes, Neutral, No)\n', to_extract_likelihoods=to_extract_likelihoods)
        # get llama predictions
        
        cond_responses, cond_probs = self.get_predictions(prompts, batch_size=batch_size, temperature=temperature, max_tokens=max_tokens, to_extract_likelihoods=to_extract_likelihoods)

        if self.cot:
            cond_responses = [c_e + '# Answer (Yes, Neutral, No)\n' + c_r for c_r, c_e in zip(cond_responses, cond_responses_expl)] 

        return cond_responses, cond_probs
    
    ## 
    # def get_probs_for_calibration(self, responses: List) -> Tuple[List[str], List[List[float]]]:
    #     log_likelihoods = []
    #     for response in responses:

    #         text = response.prompt.strip().split("\n")[-1]
            
    #         if text not in ["Yes", "No", "Neutral"]:
    #             raise ValueError(f"Invalid last token, check prompts: {text}")
            
    #         '''
    #         Modify for each LLM
    #         '''
    #         if text in ['Yes', 'No']:
    #             log_likelihood = list(response.prompt_logprobs[-1].values())[0]
    #         else:
    #             # tokenize the text
    #             tokenized_full = self.tokenizer.encode("Answer\n"+text, add_special_tokens=False) # apparently, the newline tokenizes differently
    #             answer_text = self.tokenizer.encode("Answer\n", add_special_tokens=False)
    #             tokenized_text = tokenized_full[len(answer_text):][::-1]
    #             logprobs = []
    #             tokenized_text_counter = 0
    #             for prompt_logprobs in response.prompt_logprobs[::-1]:
    #                 for token_id, logprob in prompt_logprobs.items():
    #                     if token_id == tokenized_text[tokenized_text_counter]:
    #                         logprobs.append(logprob)
    #                         tokenized_text_counter += 1
    #                     if tokenized_text_counter == len(tokenized_text):
    #                         break
    #                 if tokenized_text_counter == len(tokenized_text):
    #                     break

    #             assert len(logprobs) == len(tokenized_text)
    #             log_likelihood = sum(logprobs)
    #         log_likelihoods.append(log_likelihood)
        
    #     norm_likelihoods = []
    #     true_responses = []
    #     options = ["Yes", "No", "Neutral"]
    #     for i in range(0, len(log_likelihoods), 3:
    #         likelihoods = [math.exp(logprob) for logprob in log_likelihoods[i:i+3]]
    #         norm_likelihood = [prob/sum(likelihoods) for prob in likelihoods]
    #         # pdb.set_trace()
    #         norm_likelihoods.append(norm_likelihood)
    #         index = np.argmax(norm_likelihood)
    #         gold_response = responses[i+index].outputs[0].text
    #         true_responses.append(options[index] + gold_response)

    #     return true_responses, norm_likelihoods