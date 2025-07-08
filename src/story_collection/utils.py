from typing import Any, List, Dict
import logging
from omegaconf import DictConfig, OmegaConf
import os

logger = logging.getLogger(__name__)

class PromptBuilder:
    def __init__(self, model_name):
        self.model_name = model_name
        
    def story_generation_prompt_constructor(self, cfg: DictConfig, few_shots: Dict, background: str, label: str) -> List[Any]:
        guidelines2fewshot = {
            'yes': cfg.prompts.guidelines_affirm,
            'no': cfg.prompts.guidelines_negate,
            'neutral': cfg.prompts.guidelines_irrelevant,
        }
        prefix2fewshot = {
            'yes': os.path.join(cfg.prompts.prompts_dir, cfg.prompts.prefix_file),
            'no': os.path.join(cfg.prompts.prompts_dir, cfg.prompts.prefix_file),
            'neutral': os.path.join(cfg.prompts.prompts_dir, cfg.prompts.prefix_file).removesuffix('.txt') + '_irrelevant.txt',
        }
        guidelines_file = guidelines2fewshot[label]
        if label == 'neutral':
            prefix_file = os.path.join(cfg.prompts.prompts_dir, cfg.prompts.prefix_file).removesuffix('.txt') + '_irrelevant.txt'
            with open(f'{prefix_file}','r') as infile:
                prefix = infile.read()
        else:
            with open(f'{os.path.join(cfg.prompts.prompts_dir, cfg.prompts.prefix_file)}','r') as infile:
                prefix = infile.read()
        
        with open(f'{os.path.join(cfg.prompts.prompts_dir, guidelines_file)}','r') as infile:
            guidelines = infile.read()

        with open(f'{os.path.join(cfg.prompts.prompts_dir, cfg.prompts.reminder_file)}','r') as infile:
            reminder = infile.read()

        if self.model_name == 'gpt-3.5-turbo':
            prompt = []
            prompt.append({'role': 'user', 'content': prefix + guidelines + few_shots['user'] + background + reminder})

            # pretty-print the prompt with the role and content
            for p in prompt:
                logger.debug(f"\n[[  {p['role']}:  ]] \n{p['content']}")

        if 'gpt-4' in self.model_name:
            prompt = []
            prompt.append({'role': 'system', 'content': prefix + guidelines})
            for few_shot in few_shots:
                prompt.append({'role': 'system', 'content': few_shot['system']})
                prompt.append({'role': 'assistant', 'content': few_shot['assistant']})
            
            prompt.append({'role': 'system','content': background + reminder})

            # pretty-print the prompt with the role and content
            for p in prompt:
                logger.debug(f"\n[[  {p['role']}:  ]] \n{p['content']}")
        
        return prompt