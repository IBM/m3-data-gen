import os
import json

def extract_all_gold_question_sequences(turns):
    """Extracts all ordered sequences of questions (from full gold_sequence lists) from each turn."""
    sequences = set()
    for turn in turns:
        gold_seq = turn.get("gold_sequence", [])
        if all(isinstance(seq, dict) and "question" in seq for seq in gold_seq):
            question_sequence = tuple(seq["question"] for seq in gold_seq)
            if question_sequence:  # avoid empty tuples
                sequences.add(question_sequence)
    return sequences

def filter_by_type(input_path, sampled_output_folder, remaining_output_folder):
    with open(input_path, 'r') as f:
        data = json.load(f)

    sampled_data = []
    remaining_data = []

    # First pass: Sample based on sample_id and special types
    for item in data:
        if ("API-API-API" in item.get("type", "")) or ("RAG-API-RAG" in item.get("type", "")):
            sampled_data.append(item)
        else:
            remaining_data.append(item)

    # Collect all unique gold question sequences from sampled data
    sampled_gold_question_sequences = set()
    for item in sampled_data:
        sampled_gold_question_sequences.update(
            extract_all_gold_question_sequences(item["turns"])
        )

    # Second pass: Include items whose any gold_sequence question list matches the sampled ones
    still_remaining_data = []
    for item in remaining_data:
        item_sequences = extract_all_gold_question_sequences(item["turns"])
        if any(seq in sampled_gold_question_sequences for seq in item_sequences):
            sampled_data.append(item)
        else:
            still_remaining_data.append(item)

    # Ensure output folders exist
    os.makedirs(sampled_output_folder, exist_ok=True)
    os.makedirs(remaining_output_folder, exist_ok=True)

    # Extract filename
    filename = os.path.basename(input_path)

    # Define output paths
    sampled_output_path = os.path.join(sampled_output_folder, filename)
    remaining_output_path = os.path.join(remaining_output_folder, filename)

    # Save files
    if len(still_remaining_data) == 0:
        with open(sampled_output_path.replace("seen", "unseen"), 'w') as f:
            json.dump(sampled_data, f, indent=2)
    elif len(sampled_data) > 0:
        with open(sampled_output_path, 'w') as f:
            json.dump(sampled_data, f, indent=2)
    
    if len(still_remaining_data) > 0:
        with open(remaining_output_path, 'w') as f:
            json.dump(still_remaining_data, f, indent=2)

    print(f"Sampled {len(sampled_data)} items to {sampled_output_path}")
    print(f"Remaining {len(still_remaining_data)} items to {remaining_output_path}")




domains = ['airline', 'soccer_2016', 'beer_factory', 'movies_4', 'coinmarketcap', 'mondial_geo', 'simpson_episodes', 'hockey', 'chicago_crime', 'video_games', 'restaurant', 'olympics', 'food_inspection', 'address', 'university', 'computer_student', 'law_episode', 'movie_3', 'bike_share_1', 'college_completion', 'menu']
for domain in domains:
    filter_by_type(f"/proj/m3benchmark/raavi/api-sequence-test/balanced/domain_files/{domain}_multiturn_bird.json", "/proj/m3benchmark/raavi/api-sequence-test/balanced/test_v2/seen/domain_files/", "/proj/m3benchmark/raavi/api-sequence-test/balanced/train_v2/domain_files/")


import os
import shutil

# Define source and destination directories
src_dir = "/proj/m3benchmark/raavi/api-sequence-test/balanced/domain_files"
dst_dir = "/proj/m3benchmark/raavi/api-sequence-test/balanced/train_v2/domain_files"

# Files you don't want to move
excluded_files = {
    "/proj/m3benchmark/raavi/api-sequence-test/balanced/test/domain_files/unseen/coinmarketcap_multiturn_bird.json",
    "/proj/m3benchmark/raavi/api-sequence-test/balanced/test/domain_files/unseen/college_completion_multiturn_bird.json",
    "/proj/m3benchmark/raavi/api-sequence-test/balanced/test/domain_files/unseen/computer_student_multiturn_bird.json",
    "/proj/m3benchmark/raavi/api-sequence-test/balanced/test/domain_files/unseen/disney_multiturn_bird.json",
    "/proj/m3benchmark/raavi/api-sequence-test/balanced/test/domain_files/unseen/menu_multiturn_bird.json",
    "/proj/m3benchmark/raavi/api-sequence-test/balanced/test/domain_files/unseen/movie_multiturn_bird.json",
    "/proj/m3benchmark/raavi/api-sequence-test/balanced/test/domain_files/unseen/restaurant_multiturn_bird.json"  
}
# Just the basenames of excluded files
excluded_basenames = {os.path.basename(f) for f in excluded_files}

# Ensure destination exists
os.makedirs(dst_dir, exist_ok=True)

# Move each file
for fname in os.listdir(src_dir):
    src_path = os.path.join(src_dir, fname)
    dst_path = os.path.join(dst_dir, fname)

    # Skip non-json files
    if not fname.endswith(".json"):
        continue

    # Skip if file is excluded or already in destination
    if fname in excluded_basenames or os.path.exists(dst_path):
        continue

    # Move the file
    shutil.copy(src_path, dst_path)
    print(f"Copied: {fname}")
