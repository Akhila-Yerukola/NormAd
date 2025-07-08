from bs4 import BeautifulSoup
import requests
import pdb
import unicodedata
import logging 
import traceback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

COLOR_RED = '\033[91m'
COLOR_GREEN = '\033[92m'
COLOR_BLUE = '\033[94m'
COLOR_NORMAL = '\033[0m'

def get_base_content(axis_name, soup):
    logger.warning(f"{COLOR_RED}RETURNING RAW DATA INSTEAD!!!!\n{COLOR_NORMAL}")
    return  soup.find(
        'div', class_='text-content').text.strip(), {}


def get_core_concepts(axis_name, soup):
    content = soup.find('div', class_='text-content')
    core_values = []
    try:
        core_concept_list = soup.findAll('ul')[1]    # first list is a bunch of tabs, second one is the values
        for li in core_concept_list.findAll('li'):
            core_values.append(li.text.strip())
        core_concepts_content = {
            "core_values": core_values,
        }
    except:
        core_concepts_content = {
            "core_values": [],
        }
    # First element is some "About" section
    core_concepts_content["desc"] = content.find('p').text.strip()
    # From the second element on, we get the content for every subheading
    # The subheading itself is in the h3 tag, and the content is in the p tag
    # Get all h3 tags, then get the next sibling, which is p tags
    for h3 in content.findAll('h3'):
        core_concepts_content[h3.text.strip()] = h3.find_next_sibling('p').text.strip()
    # create a markdown prompt
    # first the list of core values
    # then the description
    # then the subheadings and content
    prompt = ""
    for core_value in core_values:
        prompt += f"- {core_value}\n"
    prompt += f"## Description  \n{core_concepts_content['desc']}\n"
    for key, value in core_concepts_content.items():
        if key != "core_values" and key != "desc":
            prompt += f"## {key}  \n{value}\n"
    # normalize the prompt string to strip out any \x characters
    prompt = prompt.encode('ascii', 'ignore').decode('ascii')
    # Could use unicodedata.normalize as well
    # problem is that some superscripts etc. also get replaced in the above method and in the one below
    # prompt = unicodedata.normalize('NFKD', prompt).encode('ascii', 'ignore').decode('ascii')
    return prompt, core_concepts_content


def get_religion(country_name, soup):
    content = soup.find('div', class_='text-content')

    religion_heads = {}
    # First element is a "Description" section
    religion_heads["desc"] = content.find('p').text.strip()
    # From the second element on, we get the content for every subheading
    # The subheading itself is in the h3 tag, and the content is in the p tag
    # Get all h3 tags, then get the next sibling, which is p tags
    for h3 in content.findAll('h3'):
        religion_heads[h3.text.strip()] = h3.find_next_sibling('p').text.strip()
    
    # create a markdown prompt
    prompt = ""

    prompt += f"## Description  \n{religion_heads['desc']}\n"
    for key, value in religion_heads.items():
        if key != "core_values" and key != "desc":
            prompt += f"## {key}  \n{value}\n"
    # normalize the prompt string to strip out any \x characters
    prompt = prompt.encode('ascii', 'ignore').decode('ascii')
    return prompt, religion_heads


def get_greetings(country_name, soup):
    get_base_content(country_name, soup)


def get_family(country_name, soup):
    get_base_content(country_name, soup)


def get_naming(country_name, soup):
    get_base_content(country_name, soup)


def get_dates(country_name, soup):
    get_base_content(country_name, soup)


def get_etiquette(country_name, soup):
    content = soup.find('div', class_='text-content')

    et_heads = {}

    et_heads_problem = False
    for h3 in content.findAll('h3'):
        # every h3 is a heading followed by a list
        # the heading is the key, the list is the value
        et_heads[h3.text.strip()] = []
        try:
            for li in h3.find_next_sibling('ul').findAll('li'):
                et_heads[h3.text.strip()].append(li.text.strip())   # strip the whitespace
        except:
            # just convert the entire thing to a string -- i.e. don't have a list
            et_heads_problem = True

    if et_heads_problem:
        logger.warning(f"{COLOR_RED}Problem with scraping Etiquette classes{COLOR_NORMAL}")
        prompt, _ = get_base_content(country_name, soup)

        # write the country name to bad_scrape.txt
        with open(f"./bad_scrape.txt", "a") as f:
            f.write(f"{country_name.strip()}\tEtiquette\n")


        et_heads['Etiquette'] = prompt
        return prompt, et_heads
    
    
    # create a markdown prompt
    prompt = ""

    for key, value in et_heads.items():
        prompt += f"## {key}  \n"
        for item in value:
            prompt += f"- {item}\n"
    # normalize the prompt string to strip out any \x characters
    prompt = prompt.encode('ascii', 'ignore').decode('ascii')

    return prompt, et_heads



def get_communication(axis_name, soup):
    get_base_content(axis_name, soup)


def get_business_culture(axis_name, soup):
    get_base_content(axis_name, soup)
