#!/bin/bash

domains=( "music_platform_2" "shooting" "car_retails" "airline" "human_resources" "student_loan" "codebase_comments" "language_corpus" "bike_share_1" "cookbook" "software_company" "donor" "authors" "shipping" "video_games" "sales" "olympics" "university" "talkingdata" "simpson_episodes" "movielens" "mondial_geo" "legislator" "regional_sales" "world_development_indicators" "food_inspection_2" "retail_world" "citeseer" "computer_student" "college_completion" "synthea" "book_publishing_company" "trains" "retails" "soccer_2016" "law_episode" "food_inspection" "european_football_1" "mental_health_survey" "hockey" "public_review_platform" "retail_complains" "ice_hockey_draft" "menu" "cs_semester" "beer_factory" "cars" "genes" "shakespeare" "image_and_language" "disney" "music_tracker" "works_cycles" "movie_platform" "books" "social_media" "restaurant" "superstore" "address" "chicago_crime" "professional_basketball" "coinmarketcap" "movies_4" "sales_in_weather" "app_store" "craftbeer" "movie" "world" )


for domain in "${domains[@]}"
do
  echo "Running: python main.py multi_turn --domain $domain --version v4 --mode gb --output_dir $OUTPUT_PATH --input_dir $INPUT_PATH"  
  python main.py multi_turn --domain $domain --version v4 --mode gb --output_dir $OUTPUT_PATH --input_dir $INPUT_PATH
  echo "Completed: multi_turn for $domain"
  echo

done
