import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm

def get_all_domains(domain_folder):
    return [name for name in os.listdir(domain_folder) if os.path.isdir(os.path.join(domain_folder, name))]

def load_and_update_samples(domain_list, domain_folder, version, balanced):
    all_samples = []
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
                print(f"Warning: {json_path} does not exist. Skipping.")
                continue
            with open(json_path, 'r') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON in {json_path}. Skipping.")
                    continue
        domain_data = []
        for sample in data:
            if "types" in sample:
                sample['type'] = sample.pop('types')
            if balanced:
                if (("RAG" in sample["type"]) or ("API-API" in sample["type"] and domain != "simpson_episodes")):
                    sample['sample_id'] = sample_counter
                    all_samples.append(sample)
                    domain_data.append(sample)
                    sample_counter += 1
            else:
                sample['sample_id'] = sample_counter
                all_samples.append(sample)
                domain_data.append(sample)
                sample_counter += 1

        os.makedirs("../api-sequence-test/balanced/domain_files", exist_ok=True)
        os.makedirs("../api-sequence-test/unbalanced/domain_files", exist_ok=True)

        if balanced:
            with open(f"../api-sequence-test/balanced/domain_files/{domain}_multiturn_bird.json", 'w') as f:
                json.dump(domain_data, f, indent=2)
        else:
            with open(f"../api-sequence-test/unbalanced/domain_files/{domain}_multiturn_bird.json", 'w') as f:
                json.dump(domain_data, f, indent=2)

    return all_samples

def main(domains_path, output_path, save_type, version, balanced):
    # domains = get_all_domains(domains_path)
    domains = [
    "address", "airline", "authors", "books", "disney", "european_football_1", "movie_platform", 
    "olympics", "professional_basketball", "university", "video_games", "world",
    "movielens", "mondial_geo", "legislator", "regional_sales", "world_development_indicators", 
    "food_inspection_2", "citeseer", "computer_student", "college_completion", 
    "synthea", "book_publishing_company", "trains", "soccer_2016", "law_episode", 
    "food_inspection", "european_football_1", "mental_health_survey", "hockey", 
    "public_review_platform", "retail_complains", "ice_hockey_draft", "menu", "cs_semester", 
    "beer_factory", "cars", "genes", "shakespeare", "image_and_language",
    "disney", "music_tracker", "works_cycles", "books", "social_media", 
    "superstore", "address", "chicago_crime", "professional_basketball", "coinmarketcap", 
    "movies_4", "app_store", "craftbeer", "movielens",
    "music_platform_2", "shooting", "car_retails", "airline", "human_resources", 
    "student_loan", "codebase_comments", "language_corpus", "cookbook", "software_company", 
    "authors", "shipping", "video_games", "sales", "olympics", "university", 
    "simpson_episodes", "sales_in_weather", "movielens", "restaurant", "retail_world"
    ]
    samples = load_and_update_samples(domains, domains_path, version, balanced)

    os.makedirs(output_path, exist_ok=True)

    save_file = os.path.join(output_path, f"{save_type}_{version}.json")

    with open(save_file, 'w') as f:
        json.dump(samples, f, indent=2)
    print(f"Saved {len(samples)} samples to {save_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--domains_folder', required=True, help="Path to folder with domain-named subfolders")
    parser.add_argument('--balanced', action='store_true', help="Create the balanced version of data")
    parser.add_argument('--output_folder', required=True, help="Path to save the output folder")
    parser.add_argument('--save_type', choices=['train', 'dev', 'seen', 'unseen', 'full'], required=True, help="Save as full.json, train.json, seen.json etc.")
    parser.add_argument('--version', default="v4", help="Version of the dataset (default: v1)")
    args = parser.parse_args()
    main(args.domains_folder, args.output_folder, args.save_type, args.version, args.balanced)
