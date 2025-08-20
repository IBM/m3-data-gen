import argparse
import json
from tqdm import tqdm
from prompt import CREATE_RAG_QUESTION
from prompt import MERGE_2_QUESTIONS, MERGE_3_QUESTIONS, ENHANCE_MULTI_TURN, MULTI_HOP_JUDGE
from llm import ChatRits
from config import mistral_config
import os
from langchain_core.messages.ai import AIMessage
from dotenv import load_dotenv
load_dotenv()

def parse_args():
    parser = argparse.ArgumentParser(description="Process JSON for merging answer fields.")
    parser.add_argument("--input_dir", type=str, required=True, help="Input folder.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save the updated JSON.")
    parser.add_argument("--domain", type=str, required=True, help="Domain name (currently unused).")
    parser.add_argument("--mode", type=str, default='local', required=True, help="Test Model -- 'local' or 'gb'")
    return parser.parse_args()

def merge_turns(turns, idx_list, chat_rits_mistral):
    if len(idx_list) == 2:
        index_1 = idx_list[0]
        index_2 = idx_list[1]
        
        Q1 = turns[index_1]['query']
        A1 = str(turns[index_1]['answer'])
        try:
            Q2 = turns[index_2]["original"]
            A2 = turns[index_2]['answer']
        except:
            return
        prompt = MERGE_2_QUESTIONS.format(Q1=Q1, A1 =A1 , Q2 = Q2)
        merged_question = chat_rits_mistral.invoke(prompt)
        if type(merged_question)==AIMessage:
            merged_question = merged_question.model_dump()["content"]
        if "<new_query>" in merged_question and "</new_query>" in merged_question:
            merged_question = merged_question.split("<new_query>")[-1].split("</new_query>")[0].strip()
        if "<new_query>" in merged_question:
            merged_question = merged_question.split("<new_query>")[-1].strip()
        if "</new_query>" in merged_question:
            merged_question = merged_question.split("</new_query>")[0].strip()
        judge_input = MULTI_HOP_JUDGE + f"""\nMerged Query: {merged_question}\nSeries of question and answers:Q1: {Q1}\nA1: {A1}\nQ2: {Q2}"""
        judge_response = chat_rits_mistral.invoke(judge_input)
        if type(judge_response)==AIMessage:
            judge_response = judge_response.model_dump()["content"]
        reasoning = judge_response.split("<reasoning>")[-1].split("</reasoning>")[0].strip()
        final_verdict = judge_response.split("<final_verdict>")[-1].split("</final_verdict>")[0].strip()
        rewritten_question = ""
        if final_verdict == "mismatch":
            rewritten_question = judge_response.split("<new_query>")[-1].split("</new_query>")[0].strip()
        turns[index_1] = {
                            "query": merged_question,
                            "answer": A2,
                            "type": "(API-API)",
                            "index": None,
                            "gold_sequence": turns[index_1]["gold_sequence"] + turns[index_2]["gold_sequence"],
                            "metadata": {"reasoning": reasoning, "final_verdict": final_verdict, "rewritten_question": rewritten_question}
                        }
        

    
    elif len(idx_list) == 3:
        index_1 = idx_list[0]
        index_2 = idx_list[1]  
        index_3 = idx_list[2]
        Q1 = turns[index_1]['query']
        A1 = str(turns[index_1]['answer'])
        try:
            Q2 = turns[index_2]["original"]
            A2 = turns[index_2]['answer']
        except:
            Q2 = turns[index_2]["query"]
            A2 = turns[index_2]['answer']
        try:
            Q3 = turns[index_3]["original"]
            A3 = turns[index_3]['answer']
            prompt = MERGE_3_QUESTIONS.format(Q1=Q1, A1 =A1 , Q2 = Q2, A2 = A2, Q3=Q3)
            merged_question = chat_rits_mistral.invoke(prompt)
            if type(merged_question)==AIMessage:
                merged_question = merged_question.model_dump()["content"]
            if "<new_query>" in merged_question and "</new_query>" in merged_question:
                merged_question = merged_question.split("<new_query>")[-1].split("</new_query>")[0].strip()
            if "<new_query>" in merged_question:
                merged_question = merged_question.split("<new_query>")[-1].strip()
            if "</new_query>" in merged_question:
                merged_question = merged_question.split("</new_query>")[0].strip()
            judge_input = MULTI_HOP_JUDGE + f"""\nMerged Query: {merged_question}\nSeries of question and answers:Q1: {Q1}\nA1: {A1}\nQ2: {Q2}\nA2: {A2}\nQ3:  {Q3}"""
            judge_response = chat_rits_mistral.invoke(judge_input)
            if type(judge_response)==AIMessage:
                judge_response = judge_response.model_dump()["content"]
            reasoning = judge_response.split("<reasoning>")[-1].split("</reasoning>")[0].strip()
            final_verdict = judge_response.split("<final_verdict>")[-1].split("</final_verdict>")[0].strip()
            rewritten_question = ""
            if final_verdict == "mismatch":
                rewritten_question = judge_response.split("<new_query>")[-1].split("</new_query>")[0].strip()

            turns[index_1] = {
                                "query": merged_question,
                                "answer": A3,
                                "type": "(API-API-API)",
                                "index": None,
                                "gold_sequence": turns[index_1]["gold_sequence"] + turns[index_2]["gold_sequence"] + turns[index_3]["gold_sequence"],
                                "metadata": {"reasoning": reasoning, "final_verdict": final_verdict, "rewritten_question": rewritten_question}
                            }
        except:
            prompt = MERGE_2_QUESTIONS.format(Q1=Q1, A1 =A1 , Q2 = Q2)
            merged_question = chat_rits_mistral.invoke(prompt)
            if type(merged_question)==AIMessage:
                merged_question = merged_question.model_dump()["content"]
            if "<new_query>" in merged_question and "</new_query>" in merged_question:
                merged_question = merged_question.split("<new_query>")[-1].split("</new_query>")[0].strip()
            if "<new_query>" in merged_question:
                merged_question = merged_question.split("<new_query>")[-1].strip()
            if "</new_query>" in merged_question:
                merged_question = merged_question.split("</new_query>")[0].strip()
            judge_input = MULTI_HOP_JUDGE + f"""\nMerged Query: {merged_question}\nSeries of question and answers:Q1: {Q1}\nA1: {A1}\nQ2: {Q2}"""
            judge_response = chat_rits_mistral.invoke(judge_input)
            if type(judge_response)==AIMessage:
                judge_response = judge_response.model_dump()["content"]
            reasoning = judge_response.split("<reasoning>")[-1].split("</reasoning>")[0].strip()
            final_verdict = judge_response.split("<final_verdict>")[-1].split("</final_verdict>")[0].strip()
            rewritten_question = ""
            if final_verdict == "mismatch":
                rewritten_question = judge_response.split("<new_query>")[-1].split("</new_query>")[0].strip()
            turns[index_1] = {
                                "query": merged_question,
                                "answer": A2,
                                "type": "(API-API)",
                                "index": None,
                                "gold_sequence": turns[index_1]["gold_sequence"] + turns[index_2]["gold_sequence"],
                                "metadata": {"reasoning": reasoning, "final_verdict": final_verdict, "rewritten_question": rewritten_question}
                            }
            idx_list = [index_1, index_2]
        



    # Remove other merged turns in reverse order to keep indices valid
    for idx in sorted(idx_list[1:], reverse=True):
        del turns[idx]

def process_json_data(data, chat_rits_mistral):
    for entry in tqdm(data):
        turns = entry.get("turns", [])
        i = 0
        while i < len(turns):
            if turns[i]["type"] != "(API)": 
                i += 1
                continue
            current = turns[i]
            answer = current.get("answer")

            if isinstance(answer, list) and len(answer) > 1:
                i += 1
                continue

            if i + 1 < len(turns):
                next_turn = turns[i + 1] # second turn
                next_answer = next_turn.get("answer")
                if next_turn["type"] != "(API)": 
                    i += 1
                    continue
                if i + 2 < len(turns) and isinstance(next_answer, list) and len(next_answer) == 1:
                    merge_turns(turns, [i, i+1, i+2], chat_rits_mistral)
                else:
                    merge_turns(turns, [i, i+1], chat_rits_mistral)

            # No merging
            i += 1
        entry["num_turns"] = len(turns)
        entry["num_hops"] = [len(turn["gold_sequence"]) for turn in entry["turns"]]
        turns_type = ""
        for turn in entry["turns"]:
            turns_type += turn["type"]
        entry["type"] = turns_type


def main():
    args = parse_args()
    try:
        with open(os.path.join(args.input_dir, f"{args.domain}_multiturn_bird.json"), "r") as f:
            data = json.load(f)
        chat_rits_mistral = ChatRits(mistral_config[args.mode])  
        process_json_data(data, chat_rits_mistral)

        with open(os.path.join(args.output_dir, f"{args.domain}_multiturn_bird.json"), "w") as f:
            json.dump(data, f, indent=2)
        print(f"Updated data saved to: {args.output_dir}")
    except:
        print("Domain not present!")

if __name__ == "__main__":
    main()
