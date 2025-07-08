import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf
import os

@hydra.main(version_base=None, config_name='config', config_path='../../conf')
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    non_cond = os.path.join(f'{cfg.story_inference.save_path}',f'etiquette_non_conditioned_{cfg.story_inference.model_name}.csv')
    value_cond = os.path.join(f'{cfg.story_inference.save_path}',f'etiquette_value_conditioned_{cfg.story_inference.model_name}.csv')
    full_cond = os.path.join(f'{cfg.story_inference.save_path}',f'etiquette_full_conditioned_{cfg.story_inference.model_name}.csv')

    non_df = pd.read_csv(non_cond,encoding='utf-8-sig')
    value_df = pd.read_csv(value_cond, encoding='utf-8-sig')
    full_df = pd.read_csv(full_cond, encoding='utf-8-sig')

    non_df = non_df.rename(columns={"Response": "non_conditioned_resp"})
    non_df['value_conditioned_resp'] = value_df['Response']
    non_df['full_conditioned_resp'] = full_df['Response']

    non_df.to_csv(os.path.join(f'{cfg.story_inference.save_path}',f'etiquette_responses_{cfg.story_inference.model_name}.csv'))

main()

