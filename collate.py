import os
import json
import argparse
from tqdm import tqdm

"""
Usage:
    PYTHONPATH=./ python collate.py --domains_folder /proj/m3benchmark/dgt/821/raw --balanced --output_folder /proj/m3benchmark/m3data --version v1
"""


def get_all_domains(domain_folder):
    return [name for name in os.listdir(domain_folder) if os.path.isdir(os.path.join(domain_folder, name))]

def load_and_update_samples(domain_list, domain_folder, output_path, red_domains, version, balanced):
    data_unbalanced=[]
    data_balanced = []
    unprocessed_domains=0
    sample_counter = 0
    for domain in tqdm(domain_list):
        try:
            json_path = os.path.join(domain_folder, domain, 'data.jsonl')
            data = []
            with open(json_path, 'r') as f:
                for line in f:
                    data.append(json.loads(line))
        except:
            json_path = os.path.join(domain_folder, domain, f'multi_turn_{version}.jsonl')
            if not os.path.exists(json_path):
                unprocessed_domains+=1
                print(f"Warning: Data for {domain} does not exist. Skipping.")
                continue
            with open(json_path, 'r') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON in {json_path}. Skipping.")
                    continue
        domain_data_balanced = []
        domain_data_unbalanced = []        
        for sample in data:
            # TODO : This fix should be moved to data generation file
            if "types" in sample:
                sample['type'] = sample.pop('types')

            # Create balanced version
            if balanced:
                if ("RAG" in sample["type"]) or ("API-API" in sample["type"]):
                # if (("RAG" in sample["type"]) or ("API-API" in sample["type"] and domain != "simpson_episodes")): # As there are more domains than simpson_episodes which have high sample size
                    sample['sample_id'] = sample_counter
                    data_balanced.append(sample)
                    domain_data_balanced.append(sample)
            sample['sample_id'] = sample_counter
            data_unbalanced.append(sample)
            domain_data_unbalanced.append(sample)
            sample_counter += 1 # Global sample counter for both balanced and unbalanced files so that these could be merged in the future

        if domain in red_domains:
            unprocessed_domains+=1
            with open(f"{output_path}/excluded_unbalanced/{domain}_multiturn_bird.json", 'w') as f:
                json.dump(domain_data_unbalanced, f, indent=2)
            continue
        if balanced:
            with open(f"{output_path}/balanced/{domain}_multiturn_bird.json", 'w') as f:
                json.dump(domain_data_balanced, f, indent=2)
        with open(f"{output_path}/unbalanced/{domain}_multiturn_bird.json", 'w') as f:
            json.dump(domain_data_unbalanced, f, indent=2)

    return data_balanced, data_unbalanced, unprocessed_domains

def main(domains_path, output_path, save_type, version, balanced):
    # domains = get_all_domains(domains_path)

    # All domains in BIRD dataset
    domains=["california_schools", "card_games", "codebase_community", "debit_card_specializing", "european_football_2", "financial", "formula_1", "student_club", "superhero", "thrombosis_prediction", "toxicology"
             ,"music_platform_2", "shooting", "car_retails", "airline", "human_resources", "student_loan", "codebase_comments", "language_corpus", "bike_share_1", "cookbook", "software_company", "donor", "authors"
             , "shipping", "video_games", "sales", "olympics", "university", "talkingdata", "simpson_episodes", "movielens", "mondial_geo", "legislator", "regional_sales", "world_development_indicators", "food_inspection_2"
             , "retail_world", "citeseer", "computer_student", "college_completion", "synthea", "book_publishing_company", "trains", "retails", "soccer_2016", "law_episode", "food_inspection", "european_football_1", "mental_health_survey"
             , "hockey", "public_review_platform", "retail_complains", "ice_hockey_draft", "menu", "cs_semester", "beer_factory", "cars", "genes", "shakespeare", "image_and_language", "disney", "music_tracker", "works_cycles"
             , "movie_platform", "books", "social_media", "restaurant", "superstore", "address", "chicago_crime", "professional_basketball", "coinmarketcap", "movies_4", "sales_in_weather", "app_store", "craftbeer", "movie", "world","movie_3"]
    
    # Domains to exclude for personal information
    red_domains=["car_retails", "synthea", "shipping", "cs_semester", "food_inspection_2"
                 , "sales", "software_company", "social_media", "human_resources", "regional_sales"
                 ,"works_cycles", "retails", "retail_world", "retail_complains", "shooting", "superstore"]

    os.makedirs(f"{output_path}/balanced/", exist_ok=True)
    os.makedirs(f"{output_path}/unbalanced/", exist_ok=True)
    os.makedirs(f"{output_path}/excluded_unbalanced/", exist_ok=True)

    samples_balanced, samples_unbalanced, unprocessed_domains = load_and_update_samples(domains, domains_path, output_path, red_domains, version, balanced)

    # save_file = os.path.join(output_path, f"{save_type}_{version}.json")

    # with open(save_file, 'w') as f:
    #     json.dump(samples, f, indent=2)
    print(f"Saved {len(samples_balanced)} data points in balanced file.")
    print(f"Saved {len(samples_unbalanced)} data points in unbalanced file.")
    print(f"Saved {len(red_domains)} domains in excluded_unbalanced folder.")
    print(f"In total {unprocessed_domains} domains were excluded.")    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--domains_folder', required=True, help="Path to folder with domain-named subfolders")
    parser.add_argument('--balanced', action='store_true', help="Create the balanced version of data")
    parser.add_argument('--output_folder', required=True, help="Path to save the output folder")
    parser.add_argument('--save_type', choices=['train', 'dev', 'seen', 'unseen', 'full'], required=False, help="Save as full.json, train.json, seen.json etc.")
    parser.add_argument('--version', default="v4", help="Version of the dataset (default: v1)")
    args = parser.parse_args()
    main(args.domains_folder, args.output_folder, args.save_type, args.version, args.balanced)
