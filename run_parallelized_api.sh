#!/bin/bash

# domains=("california_schools" "debit_card_specializing" "european_football_2" "financial" "formula_1" "student_club" "superhero" "thrombosis_prediction" "toxicology")
# domains=( "music_platform_2" "shooting" "car_retails" "airline" "human_resources" "student_loan" "codebase_comments" "language_corpus" "bike_share_1" "cookbook" "software_company" "donor" "authors" "shipping" "video_games" "sales" "olympics" "university" "talkingdata" "simpson_episodes" "movielens" "mondial_geo" "legislator" "regional_sales" "world_development_indicators" "food_inspection_2" "retail_world" "citeseer" "computer_student" "college_completion" "synthea" "book_publishing_company" "trains" "retails" "soccer_2016" "law_episode" "food_inspection" "european_football_1" "mental_health_survey" "hockey" "public_review_platform" "retail_complains" "ice_hockey_draft" "menu" "cs_semester" "beer_factory" "cars" "genes" "shakespeare" "image_and_language" "disney" "music_tracker" "works_cycles" "movie_platform" "books" "social_media" "restaurant" "superstore" "address" "chicago_crime" "professional_basketball" "coinmarketcap" "movies_4" "sales_in_weather" "app_store" "craftbeer" "movie" "world" "movie_3")
# domains=("public_review_platform")
domains=("world_development_indicators")
# domains=("simpson_episodes")
# domains=( "music_platform_2" "airline" "student_loan" "codebase_comments" "language_corpus" "bike_share_1" "cookbook" "donor" "authors" "video_games" "olympics" "university" "talkingdata" "movielens" "mondial_geo" "legislator" "citeseer" "computer_student" "college_completion" "book_publishing_company" "trains" "soccer_2016" "law_episode" "food_inspection" "european_football_1" "mental_health_survey" "hockey" "ice_hockey_draft" "menu" "beer_factory" "cars" "genes" "shakespeare" "image_and_language" "disney" "music_tracker" "movie_platform" "books" "restaurant" "address" "chicago_crime" "professional_basketball" "coinmarketcap" "movies_4" "sales_in_weather" "app_store" "craftbeer" "movie" "world" "movie_3")

MAX_JOBS=10  


run_generate_multi_turn() {
    local domain=$1
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting: python get_api_sequences.py --domain $domain --mode gb --output_dir $OUTPUT_PATH --input_dir $INPUT_PATH"
    python get_api_sequences.py --domain $domain --mode gb --output_dir $OUTPUT_PATH --input_dir $INPUT_PATH
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed: multi_turn generation for $domain"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Failed: multi_turn generation for $domain (exit code: $exit_code)"
    fi
    return $exit_code
}

# Parallel execution with job control
echo "Starting parallel execution with max $MAX_JOBS concurrent jobs..."
echo "Total domains to process: ${#domains[@]}"

for domain in "${domains[@]}"; do
    # Wait if we've reached max jobs
    while [ $(jobs -r | wc -l) -ge $MAX_JOBS ]; do
        sleep 1
    done
    
    # Run job in background
    run_generate_multi_turn "$domain" &
done

# Wait for all background jobs to complete
wait
echo "[$(date '+%Y-%m-%d %H:%M:%S')] All jobs completed!"
