from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
import omegaconf
from typing import List, Any

@dataclass
class WebScrapeConfig:
    BASE_URL: str = field()
    data_dir: str = field()
    use_axis_names: list = field(default_factory=list)

@dataclass
class PromptConfig:
    openai_key: str = field()
    prompts_dir: str = field()
    prefix_file: str = field()
    guidelines_file: str = field()
    reminder_file: str = field()
    output_dir: str = field()
    start_index: int = field(default=0)
    end_index: int = field(default=100)

@dataclass
class StoryInferenceConfig:
    path_to_story_file: str = field()
    save_path: str = field()
    model_name: str = field()
    max_tokens: int = field()
    temperature: float = field()
    no_condition: bool = field()
    value_condition: bool = field()
    full_condition: bool = field()
    do_test: bool = field(default=False)

@dataclass
class Config:
    webscrape: WebScrapeConfig = omegaconf.MISSING
    prompts: PromptConfig = omegaconf.MISSING
    story_inference: StoryInferenceConfig = omegaconf.MISSING


cs = ConfigStore.instance()
# register the config class with the name "webscrape_config"
cs.store(name="base_config", node=Config)
cs.store(group="webscrape", name="base_webscrape", node=WebScrapeConfig)
cs.store(group='prompts', name="prompts", node=PromptConfig)
