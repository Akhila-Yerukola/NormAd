import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import pdb
from src.webscrape.webscrape_utils import get_base_content, get_core_concepts, get_religion, get_greetings, get_family, get_naming, get_dates, get_etiquette, get_communication, get_business_culture
import json
import os
from dataclasses import dataclass, field
import hydra
from omegaconf import OmegaConf, DictConfig
from src.base import *

content_mapping = {
    'Core Concepts': get_core_concepts,
    'Religion': get_religion,
    'Greetings': get_greetings,
    'Family': get_family,
    'Naming': get_naming,
    'Dates of Significance': get_dates,
    'Etiquette': get_etiquette,
    'Communication': get_communication,
    'Business Culture': get_business_culture
}

def get_content(axis_name, country_name, soup):
    try:
        content_mapping[axis_name](country_name, soup)
    except KeyError:
        return get_base_content(country_name, soup)


@hydra.main(version_base=None, config_name="config", config_path="../conf")
def main(cfg: Config) -> None:
    print(OmegaConf.to_yaml(cfg))
    BASE_URL = cfg.webscrape.BASE_URL
    page = requests.get(BASE_URL + 'countries')

    soup = BeautifulSoup(page.content, "html.parser")

    country_a = soup.findAll("a", class_="text__link")

    country_text = [c.text for c in country_a]
    country_paths = [c['href'] for c in country_a]

    country_links_atlas = {}

    use_axis_names = cfg.webscrape.use_axis_names

    for cpath, country in tqdm(tuple(zip(country_paths, country_text)), initial=0, total=len(country_paths)):

        page = requests.get(BASE_URL + cpath)
        soup = BeautifulSoup(page.content, "html.parser")
        axes_content = {}
        for axis in soup.findAll('a', 'card h-100-ns'):
            axis_content = {}
            axis_name = axis.find('h2').text.strip()
            axis_url = BASE_URL + axis['href']
            page = requests.get(axis_url)
            soup = BeautifulSoup(page.content, "html.parser")
            if axis_name not in use_axis_names:
                continue
            content, content_dict = get_content(country_name=country, axis_name=axis_name, soup=soup)
            # save the content in the particular axis directory as country_name_{axis}.md
            # if the directory doesn't exist, make it
            if not os.path.isdir(f"{cfg.webscrape.data_dir}/{axis_name}"):
                os.mkdir(f"./{cfg.webscrape.data_dir}/{axis_name}")

            axis_file_name = '_'.join(axis_name.lower().strip().split())
            country_file_name = '_'.join(country.lower().strip().split())
            with open(f"{cfg.webscrape.data_dir}/{axis_name}/{country_file_name}_{axis_file_name}.md", "w") as f:
                f.write(content)
            axis_content['content'] = content_dict
            axis_content['url'] = axis_url
            # save the axis_content in the particular axis directory as country_name_axis.json
            with open(f"{cfg.webscrape.data_dir}/{axis_name}/{country_file_name}_{axis_file_name}.json", 'w') as f:
                f.write(json.dumps(axis_content, indent=4))

            axes_content[axis_name] = axis_content
            # save the axes_content in the particular country directory as country_name.json
            # if the directory doesn't exist, make it
            if not os.path.isdir(f"{cfg.webscrape.data_dir}/countries"):
                os.mkdir(f"{cfg.webscrape.data_dir}/countries")
            
            # if the file already exists, load it and update it
            if os.path.isfile(f"{cfg.webscrape.data_dir}/countries/{country_file_name}.json"):
                with open(f"{cfg.webscrape.data_dir}/countries/{country_file_name}.json", 'r') as f:
                    axes_content = json.load(f)
                    axes_content[axis_name] = axis_content

            with open(f"{cfg.webscrape.data_dir}/countries/{country_file_name}.json", 'w') as f:
                f.write(json.dumps(axes_content, indent=4))

        country_links_atlas[country] = axes_content
        # meh too big, just chuck this for now
        # with open(f"{cfg.webscrape.data_dir}/{country}/{country}.json", 'w') as f:
        #     f.write(json.dumps(country_links_atlas, indent=4))


main()
