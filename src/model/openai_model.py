from openai import OpenAI
from .base_model import BaseModel
from typing import List, Dict, Union, Any, Optional
import logging 
import traceback
from tqdm import tqdm
import pdb

logger = logging.getLogger(__name__)

COLOR_RED = '\033[91m'
COLOR_GREEN = '\033[92m'
COLOR_BLUE = '\033[94m'
COLOR_NORMAL = '\033[0m'

class OpenAIInferencer(BaseModel):
    def __init__(self, openai_api_key:str, name:str = 'gpt-3.5-turbo', chatMode:bool = True, log_batch=50):
        super().__init__()
        self.chatMode = chatMode
        self.model_name = name
        self._client = OpenAI(api_key=openai_api_key)
        self.log_batch = log_batch
    
    def __call__(self, X: List, max_tokens: Union[int, None]=None, temperature: int=0, return_probs: bool = False, batch_size: int= None):
        return self.forward(X, max_tokens,temperature, return_probs=return_probs)
    
    def forward(self, X: List, max_tokens: Union[int, None]=None, temperature: int=0, return_probs: bool = False) -> List:
        '''
        Get inference outputs from Openai model
        '''
        if return_probs:
            logger.warning(f"{COLOR_RED}return_probs is not supported for OpenAI models{COLOR_NORMAL}") 
        responses = []
        if self.chatMode:
            prompts = [{'role': 'user', 'content': x} for x in X] 
            # TODO: add logic for GPT-4 (requires different input structure)
            for ind, prompt in tqdm(enumerate(prompts), total=len(prompts)):
                try:
                    response_chat = self._client.chat.completions.create(model=self.model_name, messages= [prompt], max_tokens=max_tokens, temperature=temperature, timeout=5)
                    responses.append(response_chat.choices[0].message.content)
                except Exception as e:
                    logger.error(f"{COLOR_RED}Error for input: {prompt}{COLOR_NORMAL}") 
                    traceback.print_exc()
                    responses.append(None)
                
                if (ind+1)%self.log_batch == 0 and ind != 0:
                    with open(f'logfile_chat_{self.model_name}.txt','a') as f:
                        f.write(f"logging batches from {len(responses)-self.log_batch} to {len(responses)}\n")
                        for request, response in zip(prompts[len(responses)-self.log_batch:len(responses)], responses[-self.log_batch:]):
                            if response is None:
                                f.write(request['content']+"\t\n")
                            else:
                                f.write(request['content']+"\t"+response.strip()+"\n")
                
            with open(f'logfile_chat_{self.model_name}.txt','a') as f:
                f.write(f"logging batches from {len(responses)-((ind+1)%self.log_batch)} to {len(responses)}\n")
                for request, response in zip(prompts[-((ind+1)%self.log_batch):],responses[-((ind+1)%self.log_batch):]):
                        if response is None:
                            f.write(request['content']+"\t\n")
                        else:
                            f.write(request['content']+"\t"+response.strip()+"\n")

        else:
            prompts = X   
            for ind, prompt in tqdm(enumerate(prompts), total=len(prompts)):
                try:
                    response_text = self._client.completions.create(model=self.model_name, prompt=prompt, max_tokens=max_tokens, temperature=temperature, timeout=5)
                    responses.append(response_text.choices[0].text)
                except Exception as e:
                    logger.error(f"{COLOR_RED}Error for input: {prompt}{COLOR_NORMAL}") 
                    traceback.print_exc()
                    responses.append(None)
            
                if (ind+1)%self.log_batch == 0 and ind != 0:
                    with open(f'logfile_text_{self.model_name}.txt','a') as f:
                        f.write(f"logging batches from {len(responses)-self.log_batch} to {len(responses)}\n")
                        for request, response in zip(prompts[-self.log_batch:len(responses)], responses[-self.log_batch:]):
                            if response is None:
                                f.write(request+"\t\n")
                            else:
                                f.write(request+"\t"+response.strip()+"\n")
                
            with open(f'logfile_text_{self.model_name}.txt','a') as f:
                f.write(f"logging batches from {len(responses)-((ind+1)%self.log_batch)} to {len(responses)}\n")
                for request, response in zip(prompts[-((ind+1)%self.log_batch):],responses[-((ind+1)%self.log_batch):]):
                        if response is None:
                            f.write(request+"\t\n")
                        else:
                            f.write(request+"\t"+response.strip()+"\n")
        
        return responses, None
              
        

        
        
        

