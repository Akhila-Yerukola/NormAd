import pandas as pd
import os
import hydra 
from omegaconf import DictConfig
import pdb
def get_csv_from_responses(cfg: DictConfig, axis: str ='Etiquette') -> None:
    # list the output files
    files_path = os.path.join(cfg.prompts.output_dir,axis)
    filenames = os.listdir(files_path)
    
    axis_name = "_".join(axis.lower().strip().split())
    resp = []
    for file in filenames:
        # find the index of the axis
        country = "_".join(file[:file.find(axis_name)].split('_')[:-1])
        subaxes = "_".join(file[file.find(axis_name):-3].split('_')[1:]) # strip .md
        # get the content
        with open(os.path.join(files_path,file),'r') as f:
            content = f.read()
        # extract Value: Rule of Thumb: Story: Explanation:
        value_index = content.lower().rindex('value:')+len('Value:')
        rule_index = content.lower().rindex('rule-of-thumb:')+len('Rule-of-Thumb:')
        story_index = content.lower().rindex('story:')+len('Story:')
        explanation_index = content.lower().rindex('explanation:')+len('Explanation:')
        
        # get the values, and strip any ##
        value = content[value_index:rule_index-len('Rule-of-Thumb:')].strip().strip('#').strip() if rule_index > value_index else content[value_index:].strip().strip('#').strip()
        rule = content[rule_index:story_index-len('Story:')].strip().strip('#').strip() if story_index > rule_index else content[rule_index:].strip().strip('#').strip()
        story = content[story_index:explanation_index-len('Explanation:')].strip().strip('#').strip() if explanation_index > story_index else content[story_index:].strip().strip('#').strip()
        explanation = content[explanation_index:].strip().strip('#').strip() if explanation_index > story_index else ""
        country_background = content.find(f"### Cultural background [{axis}]:")+len(f"### Cultural background [{axis}]:")
        country_background_end = content.find("### NOTE")
        country_background = content[country_background:country_background_end].strip().strip('#').strip()
        # add to a json
        resp.append({'Country': country, 'Background': country_background, 'Axis': axis_name ,'Subaxes': subaxes, 'Value': value, 'Rule-of-Thumb': rule, 'Story': story, 'Explanation': explanation})
    
    # create a dataframe with the columns as "Country","Subaxes", "Value", "Rule of Thumb", "Story", "Explanation"
    df = pd.DataFrame(resp)
    df.to_csv(os.path.join(cfg.prompts.output_dir, f"{axis}.csv"), index=False)


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    get_csv_from_responses(cfg=cfg, axis='Etiquette')

if __name__ == '__main__':
    main()