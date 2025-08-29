import zipfile
import argparse
import json
from collections import defaultdict
import os
import tempfile
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from utils import ChatRits

MODEL_NAME = "mistralai/Mixtral-8x22B-Instruct-v0.1"

CREATE_DESCRIPTION = """
Now you are assistant who generates a description for a document retriever. You are given a table names and column descriptions below.\n
1. Use these to generate a coherent descriptions without adding any extra information from your end.
2. Make the desciption feel like a natural description without using explicit table names.
3. You are generating the description for domain {db_id}
4. Please keep the descriptions within 2-3 sentences. These are being used as tool descriptions.

The table names and column descriptions :
{text}
"""
DATASET_FOLDERNAME="/proj/m3benchmark/bird-train/raw/"

"""
Script to generate descriptions for retrievers. 
USAGE:
PYTHONPATH=./ python retrievers/generate_retriever_descriptions.py
"""

def get_col_descriptions(dataset_foldername):
    col_description=defaultdict(defaultdict)
    zip_path=dataset_foldername+'/train_databases.zip'
    archive = zipfile.ZipFile(dataset_foldername+'/train_databases.zip', 'r')
    filelist=[f for f in archive.namelist() if (".csv" in f) and ("_MACOSX" not in f)]
    for filename in filelist:
        db_id, table_name=filename.split("/")[1], filename.split("/")[3].split(".csv")[0]
        db_path = filename
        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                extracted_path= zip_ref.extract(db_path, temp_dir)
                try:
                    df = pd.read_csv(extracted_path)
                except:
                    print(f"{extracted_path} didn't work")
                    continue
        descriptions=[str(i) for i in df["column_description"].tolist()]
        col_description[db_id][table_name]=" ; ".join(descriptions)
    return col_description

def strfromdict(dict_obj):
    str_obj=""
    for k, v in dict_obj.items():
        str_obj=str_obj+f"{k} : {v}\n"
    return str_obj

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_foldername', required=False, default=DATASET_FOLDERNAME, help="Dataset folder having the train_databases.zip")
    parser.add_argument('--output_filename', required=False ,default="m3-data-gen/retrievers/retriever_descriptions.json", help="Path to save the output file.")
    args = parser.parse_args()

    llm = ChatRits() # Set RITS_API_KEY for this to run
    data = get_col_descriptions(args.dataset_foldername)

    domain_descriptions={}
    for k, v in tqdm(data.items()):
        prompt=CREATE_DESCRIPTION.replace("{text}", strfromdict(v)).replace("{db_id}",k)
        response=llm.invoke(prompt)
        print(f"{k} : {response.content}")
        domain_descriptions[k]=response.content    

    with open(args.output_filename, "w") as f:
        json.dump(domain_descriptions, f)