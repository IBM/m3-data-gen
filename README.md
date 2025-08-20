# multi-turn-multi-hop
Repository for creating multi-turn multi-hop queries 
For each domain, the following four commands are run in sequence. Replace the placeholders with your actual values.
First, install the dependencies using 
```bash
pip install -r requirements.txt
```
Then the code is structured into 4 parts (the sample usage of all four parts is also present in `run_ccc.sh`):
## Extract Answers
```bash
python main.py extract_answers --domain <DOMAIN> --database <DATABASE> --db_folder <DB_FOLDER> --json_file <JSON_FILE> --mode <MODE>
```

Parameters

`--domain` : Identifier for the domain being processed.

`--database` : Name of the target database (bird/spider).

`--db_folder` : Path to the directory containing the database files (path to the database in your local machine).

`--json_file` : Path to the input JSON file containing the questions.

`--mode` : Execution mode — typically local or gb [use local only].

### Sample command:
```bash
python main.py extract_answers --domain disney --database bird --db_folder /proj/m3benchmark/bird-train/raw/train_databases --json_file /proj/m3benchmark/bird-train/raw/train.json --mode local
```


## Knowledge Graph Generation
```bash
python main.py knowledge_graph --base_directory <BASE_DIRECTORY> --domain <DOMAIN> --entity_directory <ENTITY_DIRECTORY> --db_folder <DB_FOLDER> --mode <MODE>
```

Parameters

`--base_directory`: Path where the output files will be stored (output/bird).

`--domain`: Identifier for the domain being processed.

`--entity_directory`: Path to the entity file (wikidata5m_entity.txt).

`--db_folder`: Path to the directory containing the database files.

`--mode`: Execution mode — typically local or gb.

### Sample command:
```bash
python main.py knowledge_graph --base_directory ./output/bird --domain disney --entity_directory ./wikidata5m_entity.txt --db_folder /proj/m3benchmark/bird-train/raw/train_databases --mode local
```

## Knowledge graph vizualization
```bash
python main.py vizualize_links --json_file <BASE_DIRECTORY>/<DOMAIN>/<DOMAIN>.json --mode <MODE>
```

### Sample command:
```bash
python main.py vizualize_links --json_file /proj/m3benchmark/bird-train/output/bird/disney/disney.json --mode local
```

Parameters

`--json_file`: Path to the JSON file generated during knowledge graph construction.

`--mode`: Execution mode — typically local.

> **Note:** The above three steps are meant to be done on local since they don't require any language model usage.

## Multi-turn dialogue creation

### Using RITS (non mistral-large model)
```bash
export RITS_API_KEY=<your_api_key>
python main.py multi_turn --domain <DOMAIN> --version <VERSION>
```

#### Sample command:
```bash
python main.py multi_turn --domain disney --version local
```

### Using granite.build 

Install gbcli by following the instructions in this [repo](https://github.ibm.com/granite-dot-build/gbcli).
A sample PR is present at [this pull](https://github.ibm.com/granite-dot-build/gbspace-public/pull/5591)

#### Uploading the genereated graphs to lakehouse [Optional]
```bash
gb artifact push --from-local <path to output/ dir> --artifact-name multi_turn_input_files --type fileset --certify-no-restrictions
```
The output of the following command will provide a uri which has to be updated in multi_turn_granite_build/ build.yaml (line 8).

#### Using the already generated graphs
The graphs currently are present at: lh://prod/granite_dot_build.public/filesets/fileset_shared/multi_turn/20250723T165326.
The generated graphs as well as the input bird data is present in lakehouse in the link before and can also be viewed on CCC at `/proj/m3benchmark/bird-train/output/bird`
```bash
cd multi_turn_granite_build
gb build start
```

Once the run completes, the results can be downloaded via:
```bash
gb artifact download --artifact-id <artifact-id>
```

Further post-processing is required to get API-API-API and API-API queries too.

#### Using the already generated multi turn conversations
```bash
cd api_sequences_granite_build
gb build start
```
#### Using the newly created multi turn conversations

First upload the multi-turn conversations downloaded using:
```bash
gb artifact push --from-local <path to conversations> --artifact-name multi_turn_files --type fileset --certify-no-restrictions
```
Update the uri obtained in api_sequences_granite_build/build.yaml
```
cd api_sequences_granite_build
gb build start
```

### Using DGT
Clone the [repo](https://github.ibm.com/DGT/fms-dgt/tree/m3_databuilders) and install the dependencies mentioned. Note, you also need to have the granite.build env (can be installed by following this [repo](https://github.ibm.com/granite-dot-build/gbcli). Let's call this environment gb.)
Head to fms_dgt repository.
```
conda activate gb
git checkout m3_databuilders
cd fms_dgt/research/databuilders/m3
gb build start
```

The multi-turn conversations will be generated using this!

## Collating the results and generating the final train-test-split:

1. Update the domains in `collate.py` which you want to focus on.
Run `collate.py`

Required Arguments:

`--domains_folder`: Path to the main folder containing subfolders for each domain.

`--output_folder`: Directory where the output JSON file will be saved.

`--save_type`: Choose one of the following to specify the file name/type (train, seen, unseen):

`--version`: Version tag to label the output.

Optional Flags
`--balanced`: If included, generates a balanced version of the dataset.

Balanced dataset: `("RAG" in sample["type"]) or ("API-API" in sample["type"] and domain != "simpson_episodes")`
If there is a RAG query (either present in a hop or as is in the dialogue) or there is an API-API question present in any domain apart from simpson_episodes (due to overflooding of simpson episodes), then that dialogue is added to the final sampled dataset.

### Sample command:
```bash
python collate.py --domains_folder /proj/m3benchmark/raavi/api-sequence-test/balanced/test_v2/unseen/domain_files --balanced --output_folder /proj/m3benchmark/raavi/api-sequence-test/balanced/test_v2/unseen --save_type unseen --version v7
```

## Train-test splits
Line 76 needs to be changed in to account for the previous output folder that you put in.
Similarly lines 83 and 84 need to be changed.
The test domains (present in line 88-94) need to be copied manually.
```bash
python test_splits.py
```

Current train-test splits are present at:
1. train: `/proj/m3benchmark/bird-train/multi_turn/train_v7_0806`
2. test: `/proj/m3benchmark/bird-dev/multi_turn/test_v7_0806`

## Getting the stats

Point the input json present in line 7 to the collated file for which you want to generate stats for. The plot dir and the version can be anything of your choice.

```bash
python stats.py
```







