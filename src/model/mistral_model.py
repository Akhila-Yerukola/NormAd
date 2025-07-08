from vllm import LLM, SamplingParams
from typing import List, Union
import logging 
import traceback
import pdb
from .base_model import BaseModel
from fastchat.conversation import Conversation, SeparatorStyle

logger = logging.getLogger(__name__)

COLOR_RED = '\033[91m'
COLOR_GREEN = '\033[92m'
COLOR_BLUE = '\033[94m'
COLOR_NORMAL = '\033[0m'

class MistralInferencer(BaseModel):
    def __init__(self, model_name_or_path='mistralai/Mistral-7B-v0.1', tensor_parallel_size=2, max_tokens=512):
        self.llm = LLM(model_name_or_path, tensor_parallel_size=tensor_parallel_size)
        self.max_tokens = max_tokens
    def __call__(self, X: List, max_tokens: Union[int, None]=None, temperature: int=0):
        return self.forward(X, max_tokens,temperature)


    def forward(self, X: List, max_tokens: Union[int, None]=None, temperature: int=0) -> List:
        '''
        Get inference outputs from Llama model
        ''' 
        responses = []
        custom_template = Conversation(
            name="mistral",
            system_template="[INST] {system_message}\n",
            roles=("[INST]", "[/INST]"),
            sep_style=SeparatorStyle.LLAMA2,
            sep=" ",
            sep2="\n",
        )
        prompts = self.create_instruct_prompts(X, 'mistral', custom_template=custom_template, assistant_prefix="# Answer")

        sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        responses = self.llm.generate(prompts, sampling_params=sampling_params)
        
        responses = [r.outputs[0].text for r in responses]
        return responses