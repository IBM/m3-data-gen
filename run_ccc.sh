#!/bin/bash

log () {
    d=`date -u '+%Y-%m-%dT%H:%M:%S.%3NZ'`
    echo $d 'run_train.sh [INFO ]:' $*
}
logLine () {
    log '---------------------------------------------------------------------------'
}
logHeader () {
    logLine
    log $*
    logLine
}

logHeader 'Started running'
runStart=$(date +%s)

set -e
set -x

export PYTHONPATH=.

RUN_DATE=$(date "+%m%d")
mkdir -p runs/${RUN_DATE}

DB_ID=("music_platform_2" "shooting" "car_retails" "airline" "human_resources" "student_loan" "codebase_comments" "language_corpus" "bike_share_1" "cookbook" "software_company" "donor" "authors" "shipping" "video_games" "sales" "olympics" "university" "talkingdata" "simpson_episodes" "movielens" "mondial_geo" "legislator" "regional_sales" "world_development_indicators" "food_inspection_2" "retail_world" "citeseer" "computer_student" "college_completion" "book_publishing_company" "trains" "retails" "soccer_2016" "law_episode" "food_inspection" "european_football_1" "mental_health_survey" "hockey" "public_review_platform" "retail_complains" "ice_hockey_draft" "menu" "cs_semester" "beer_factory" "cars" "genes" "shakespeare" "image_and_language" "disney" "music_tracker" "works_cycles" "movie_platform" "books" "social_media" "restaurant" "superstore" "address" "chicago_crime" "professional_basketball" "coinmarketcap" "movies_4" "sales_in_weather" "app_store" "craftbeer" "movie" "world" "movie_3")

VERSION_NUM=1

# Cluster resources
CLUSTER_CMD='bsub -n 1 -R "rusage[mem=20GB, cpu=4]"'

# Paths
database="bird"
db_folder="../train/train_databases"
json_file="../train/train.json"
base_directory="./output/bird "
entity_directory="./wikidata5m_entity.txt"

for id in "${DB_ID[@]}"; do
    script_file="runs/${RUN_DATE}/job_${id}.sh"
    log_file="runs/${RUN_DATE}/job_${id}.log"

    cat <<EOF > $script_file
#!/bin/bash
set -e
set -x
export PYTHONPATH=.

python main.py extract_answers --domain $id --database $database --db_folder $db_folder --json_file $json_file --mode local
python main.py knowledge_graph --base_directory $base_directory --domain $id --entity_directory $entity_directory --db_folder $db_folder --mode local
python main.py vizualize_links --json_file $base_directory/$id/$id.json --mode local
python main.py multi_turn --domain $id --version v2
EOF

    chmod +x "$script_file"

    bsub -n 1 -R "rusage[mem=20GB, cpu=4]" -J "db_${id}" -o "$log_file" bash "$script_file"

    logHeader "Submitted job for DB: $id"
    sleep 0.5
done


logHeader "All jobs submitted"
