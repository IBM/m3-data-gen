import json
import random
import re
import requests
from tqdm import tqdm
from prompt import CREATE_RAG_QUESTION
from prompt import MERGE_2_QUESTIONS, MERGE_3_QUESTIONS, ENHANCE_MULTI_TURN, MULTI_HOP_JUDGE
from llm import ChatRits
from config import mistral_config
from langchain_core.messages.ai import AIMessage
random.seed(42)
NUM_HOPS = [1, 2, 3]
MAX_NUM_TURNS = 7
NEGATIVE_WORD = "CAN'T GENERATE"
ENHANCE_MULTI_TURN_EXAMPLE = """[
  { "query": "What movie did the director of Pinocchio go on to be an executive producer for?", "answer": "Alice in Wonderland" },
  { "query": "Who was the voice actor of the best friend of Mad Hatter in the movie 'Alice in Wonderland'?" }
]"""

    
class MultiTurn:
    def __init__(self, domain, input_json, label_cache_file, chat_rits_mistral):
        try:
            with open(input_json, 'r') as file:
                self.data = json.load(file)
        except Exception as e:
            print(f"Unable to find json file: {e}")
        
        try:
            with open(label_cache_file, 'r') as file:
                self.label_cache = json.load(file)
        except Exception as e:
            print(f"Unable to find json file: {e}")
        
        self.domain = domain
        self.chat_rits_mistral = chat_rits_mistral

    def club_rag_e1_e2(self):
        e1_e2 = {}
        for element in self.data:
            if not element['links']:
                continue
            for answer in element['links']:
                for r_e2 in element['links'][answer]:
                    key = f"{len(answer)}:{answer}{len(r_e2[0])}:{r_e2[0]}"
                    if key not in e1_e2:
                        e1_e2[key] = set()
                    e1_e2[key].add(r_e2[1])
        
        e1_e2_serializable = {k: list(v) for k, v in e1_e2.items()}
        return e1_e2_serializable
    
    def extract_paragraphs_with_entity_exact(self, page_content, entity):
        paragraphs = [p.strip() for p in page_content.split('\n\n') if p.strip()]
        pattern = re.compile(rf'\b{re.escape(entity)}\b', re.IGNORECASE)
        matching_paragraphs = [p for p in paragraphs if pattern.search(p)]
        return matching_paragraphs
    
    def get_wikipedia_content_from_qid(self, qid, language='en'):
        wikidata_url = 'https://www.wikidata.org/w/api.php'
        params = {
            'action': 'wbgetentities',
            'ids': qid,
            'format': 'json',
            'props': 'sitelinks',
        }

        response = requests.get(wikidata_url, params=params)
        data = response.json()

        try:
            sitelinks = data['entities'][qid]['sitelinks']
            wiki_key = f'{language}wiki'
            title = sitelinks[wiki_key]['title']
        except KeyError:
            return f"No {language} Wikipedia page found for {qid}"

        wikipedia_url = 'https://en.wikipedia.org/w/api.php'
        params = {
            'action': 'query',
            'format': 'json',
            'prop': 'extracts',
            'explaintext': 1,
            'titles': title,
        }

        response = requests.get(wikipedia_url, params=params)
        content_data = response.json()

        # Extract content
        pages = content_data['query']['pages']
        page = next(iter(pages.values()))
        return page.get('extract', 'No content found.')
    
    def create_rag_only_questions(self, relationships):
        rag_questions = {}
        for e1_r in tqdm(relationships, total=len(relationships), desc="Creating RAG only questions!"):
            first_colon = e1_r.index(':')
            len_e1 = int(e1_r[:first_colon])
            e1_start = first_colon + 1
            e1_end = e1_start + len_e1
            e1 = e1_r[e1_start:e1_end]

            second_part = e1_r[e1_end:]
            second_colon = second_part.index(':')
            len_r = int(second_part[:second_colon])
            r_start = e1_end + second_colon + 1
            r_end = r_start + len_r
            r = e1_r[r_start:r_end]

            for e2 in relationships[e1_r]:           
                document1 = ""
                document2 = ""
                content = self.get_wikipedia_content_from_qid(self.label_cache[e1])
                matches = self.extract_paragraphs_with_entity_exact(content, e2)
                if matches:
                    for match in matches:
                        document1 += f"\n{match}"
                else:
                    document1 = ""
                
                content = self.get_wikipedia_content_from_qid(self.label_cache[e2])
                matches = self.extract_paragraphs_with_entity_exact(content, e1)
                if matches:
                    for match in matches:
                        document2 += f"\n{match}"
                else:
                    document2 = ""

                prompt = CREATE_RAG_QUESTION.format(e1=e1, e2=e2, document1=document1, document2=document2)
                rag_question = self.chat_rits_mistral.invoke(prompt)
                if type(rag_question)==AIMessage:
                    rag_question = rag_question.model_dump()["content"]
                rag_questions[f"{len(e1)}:{e1}{len(r)}:{r}{len(e2)}:{e2}"] = {"query": rag_question, "answer": e2, "documents": [document1, document2]}
        return rag_questions


    def create_api_before_rag_questions(self, rag_questions):
        output_data = []
        sample_id = 0
        for id, element in tqdm(enumerate(self.data), desc="Generating API before RAG questions!", total=len(self.data)):
            if len(element['answer']) == 1 and 'links' in element:
                prompt = MERGE_2_QUESTIONS.format(Q1 = element['question'], A1 = str(element['answer']))
                for answer in element['links']:
                    for relations in element['links'][f'{answer}']:
                        r = relations[0]
                        e2 = relations[1]
                        relation = [answer, r, e2]
                        if NEGATIVE_WORD in rag_questions[self.encode(relation)]["query"]:
                            continue
                        relation_prompt = prompt.format(Q2=rag_questions[self.encode(relation)]["query"])
                        merged_question = self.chat_rits_mistral.invoke(relation_prompt)
                        if type(merged_question)==AIMessage:
                            merged_question = merged_question.model_dump()["content"]
                        merged_question = merged_question.replace("### Merged Question", "")
                        match = re.search(r"<new_query>(.*?)</new_query>", merged_question)
                        if match:
                            merged_question = match.group(1)
                        turns = [{"query":merged_question, "metadata":  [element, relation], "answer":rag_questions[self.encode(relation)]["answer"], "gold_sequence": [element, rag_questions[self.encode(relation)]["documents"]], "index": None, "type": "(API-RAG)"}]
                        output = {
                            "sample_id": sample_id,
                            "dataset_name": self.domain,
                            "turns": turns,
                            "type": "(API-RAG)",
                            "num_turns": 1,
                            "num_hops": [2],
                            "tools": None 
                        }
                        output_data.append(output)
                        sample_id += 1
        return output_data

    
    def create_rag_before_api_questions(self, rag_questions):
        output_data = []
        sample_id = 0
        for id, element in tqdm(enumerate(self.data), total=len(self.data)):
            if 'links' in element:
                for answer in element['links']:
                    e1 = answer
                    for r_e2_qs in element['links'][f"{e1}"]:
                        r = r_e2_qs[0]
                        e2 = r_e2_qs[1]
                        relation = [e1, r, e2]
                        if NEGATIVE_WORD in rag_questions[self.encode(relation)]["query"]:
                            continue
                        for question_id in r_e2_qs[2]:
                            prompt = MERGE_2_QUESTIONS.format(Q1 = rag_questions[self.encode(relation)]["query"], A1 = rag_questions[self.encode(relation)]["answer"])
                            relation_prompt = prompt.format(Q2 = self.data[question_id]['question'])
                            merged_question = self.chat_rits_mistral.invoke(relation_prompt)
                            if type(merged_question)==AIMessage:
                                merged_question = merged_question.model_dump()["content"]
                            merged_question = merged_question.replace("### Merged Question", "")
                            match = re.search(r"<new_query>(.*?)</new_query>", merged_question)
                            if match:
                                merged_question = match.group(1)
                            turns = [{"query":merged_question, "metadata":  [relation, self.data[question_id]], "answer":self.data[question_id]["answer"], "gold_sequence": [rag_questions[self.encode(relation)]["documents"], self.data[question_id]], "index": None, "type": "(RAG-API)"}]
                            output = {
                                "sample_id": sample_id,
                                "dataset_name": self.domain,
                                "turns": turns,
                                "type": "(RAG-API)",
                                "num_turns": 1,
                                "num_hops": [2],
                                "tools": None 
                            }
                            output_data.append(output)
                            sample_id += 1
        return output_data
    
    def encode(self, entities):
        output = ""
        for entity in entities:
            output += f"{len(entity)}:{entity}"
        return output
    
    def get_outgoing_questions_rag(self, question_id, rag_questions):
        question_ids = {}
        if len(self.data[question_id]['answer']) == 1 and 'links' in self.data[question_id]:
            for answer in self.data[question_id]['links']:
                for r_e2_qs in self.data[question_id]['links'][f"{answer}"]:
                    if NEGATIVE_WORD in rag_questions[self.encode([answer, r_e2_qs[0], r_e2_qs[1]])]["query"] or r_e2_qs[1] in self.data[question_id]["parameters"]:
                        continue
                    for outgoing_question in r_e2_qs[2]:
                        if outgoing_question not in question_ids:
                            question_ids[outgoing_question] = []
                        question_ids[outgoing_question].append([answer, r_e2_qs[0], r_e2_qs[1]])

        if self.data[question_id]['direct_links']:
            for outgoing_question in set(self.data[question_id]['direct_links']):
                if outgoing_question != question_id and outgoing_question not in question_ids:
                    question_ids[outgoing_question] = []
                if outgoing_question != question_id:
                    question_ids[outgoing_question].append([])
        
        return question_ids
    
    
    def dfs_rag(self, question_id, visited, curr_path, paths, ere, rag_questions):
        if visited[question_id]:
            return
        if len(ere) == 0:
            curr_path.append(question_id)
        else:
            curr_path.append(ere)  
            curr_path.append(question_id)
        visited[question_id] = True
        outgoing_questions = self.get_outgoing_questions_rag(question_id, rag_questions)
        if len(outgoing_questions) == 0:
            paths.append(curr_path.copy())
        else:
            for question in outgoing_questions:
                self.dfs_rag(question, visited, curr_path, paths, outgoing_questions[question], rag_questions)
        curr_path.pop()  
        if ere:
            curr_path.pop()  


    def load_qid_text_mapping(self, file_path):
        qid_to_text = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if '\t' in line:
                    qid, text = line.strip().split('\t', 1)
                    qid_to_text[qid] = text
        return qid_to_text
    
    def chunk_list(self, lst, chunk_size=8):
        return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]
            
    def run(self, rag_questions):
        input_paths = []
        curr_path = []
        for id, element in enumerate(self.data):
            if 'links' in element or 'direct_links' in element:
                visited = [False] * len(self.data)
                self.dfs_rag(id, visited, curr_path, input_paths, [], rag_questions)
        paths = []
        for path in input_paths:
            if path[-1] == [[]]:
                path = path[:-1]
            if path not in paths and len(path) > 1:
                paths.append(path)
        multi_turn = []
        print(f"Total multi-turn queries: {len(paths)}")
        for i, path in tqdm(enumerate(paths), desc="Turning paths into multi-turn queries!", total=len(paths)):
            sample = {}
            sample["sample_id"] = i
            sample["dataset_name"] = self.domain
            turns = []
            hops = []
            start_index = 0
            max_index = len(path) - 1
            types = ""
            while start_index <= max_index:
                num_hops = min(random.choices(NUM_HOPS, weights=(10, 60, 30), k=1)[0], max_index - start_index + 1)
                if num_hops == 1:
                    if isinstance(path[start_index], int):
                        gold_sequence = self.data[path[start_index]]
                        gold_sequence["question_type"] = "API"
                        turns.append({
                            "query": self.data[path[start_index]]['question'],
                            "answer": self.data[path[start_index]]['answer'],
                            "type": "(API)",
                            "gold_sequence": [gold_sequence],
                            "index": None,
                            "metadata": None
                        })
                        types += "(API)"
                        start_index += num_hops
                        hops.append(num_hops)
                    else:
                        relation = random.choice(path[start_index])
                        if relation:
                            rag_question = rag_questions[self.encode(relation)]["query"]
                            gold_sequence = [{"question": rag_question, "answer": rag_questions[self.encode(relation)]["answer"], "relation": relation, "rag_doc": rag_questions[self.encode(relation)]["documents"], "question_type": "RAG", "db_id": self.domain}]
                            turns.append({
                                "query": rag_question,
                                "answer": rag_questions[self.encode(relation)]["answer"],
                                "type": "(RAG)",
                                "gold_sequence": gold_sequence,
                                "index": None,
                                "metadata": None
                            })
                            types += "(RAG)"
                            start_index += num_hops
                            hops.append(num_hops)

                        else:
                            start_index += num_hops
                            continue
                            


                elif num_hops == 2:
                    if start_index + 1 > max_index:
                        continue
                    if isinstance(path[start_index], int):
                        #API_BEFORE_RAG
                        relation = random.choice(path[start_index + 1])
                        if relation:
                            e1 = relation[0]
                            r = relation[1]
                            e2 = relation[2]
                            if len(self.data[path[start_index]]['answer']) < 2:
                                Q1 = self.data[path[start_index]]['question']
                                A1 = str(self.data[path[start_index]]['answer'])
                                Q2 = rag_questions[self.encode(relation)]["query"]
                                prompt = MERGE_2_QUESTIONS.format(Q1=Q1, A1 =A1 , Q2 = Q2)
                                merged_question = self.chat_rits_mistral.invoke(prompt)
                                if type(merged_question)==AIMessage:
                                    merged_question = merged_question.model_dump()["content"]
                                merged_question = merged_question.replace("### Merged Question", "")
                                if "<new_query>" in merged_question and "</new_query>" in merged_question:
                                    merged_question = merged_question.split("<new_query>")[-1].split("</new_query>")[0].strip()
                                if "<new_query>" in merged_question:
                                    merged_question = merged_question.split("<new_query>")[-1].strip()
                                if "</new_query>" in merged_question:
                                    merged_question = merged_question.split("</new_query>")[0].strip()
                                judge_input = MULTI_HOP_JUDGE + f"""\nMerged Query: {merged_question}\nSeries of question and answers:Q1: {Q1}\nA1: {A1}\nQ2: {Q2}"""
                                judge_response = self.chat_rits_mistral.invoke(judge_input)
                                if type(judge_response)==AIMessage:
                                    judge_response = judge_response.model_dump()["content"]
                                reasoning = judge_response.split("<reasoning>")[-1].split("</reasoning>")[0].strip()
                                final_verdict = judge_response.split("<final_verdict>")[-1].split("</final_verdict>")[0].strip()
                                rewritten_question = ""
                                if final_verdict == "mismatch":
                                    rewritten_question = judge_response.split("<new_query>")[-1].split("</new_query>")[0].strip()

                                gold_sequence_api = self.data[path[start_index]]
                                gold_sequence_api["question_type"] = "API"
                                gold_sequence_rag = {"question": rag_questions[self.encode(relation)]["query"], "answer": rag_questions[self.encode(relation)]["answer"], "relation": relation, "rag_doc": rag_questions[self.encode(relation)]["documents"], "question_type": "RAG", "db_id": self.domain}
                                turns.append({
                                    "query": merged_question,
                                    "answer": rag_questions[self.encode(relation)]["answer"],
                                    "type": "(API-RAG)",
                                    "gold_sequence": [gold_sequence_api, gold_sequence_rag],
                                    "index": None,
                                    "metadata": {"reasoning": reasoning, "final_verdict": final_verdict, "rewritten_question": rewritten_question}
                                })
                                types += "(API-RAG)"
                                start_index += num_hops
                                hops.append(num_hops)
                        else:
                            gold_sequence_api = self.data[path[start_index]]
                            gold_sequence_api["question_type"] = "API"
                            turns.append({
                                "query": self.data[path[start_index]]['question'],
                                "answer": self.data[path[start_index]]['answer'],
                                "type": "(API)",
                                "gold_sequence": [gold_sequence_api],
                                "index": None
                            })
                            types += "(API)"
                            start_index += num_hops
                            hops.append(1)
                    else:
                        relation = random.choice(path[start_index])
                        if relation:
                            e1 = relation[0]
                            r = relation[1]
                            e2 = relation[2]
                            Q1 = rag_questions[self.encode(relation)]["query"]
                            A1 = rag_questions[self.encode(relation)]["answer"]
                            Q2 = self.data[path[start_index + 1]]['question']
                            prompt = MERGE_2_QUESTIONS.format(Q1=Q1, A1 =A1, Q2=Q2)
                            merged_question = self.chat_rits_mistral.invoke(prompt)
                            if type(merged_question)==AIMessage:
                                merged_question = merged_question.model_dump()["content"]
                            if "<new_query>" in merged_question and "</new_query>" in merged_question:
                                merged_question = merged_question.split("<new_query>")[-1].split("</new_query>")[0].strip()
                            if "<new_query>" in merged_question:
                                merged_question = merged_question.split("<new_query>")[-1].strip()
                            if "</new_query>" in merged_question:
                                merged_question = merged_question.split("</new_query>")[0].strip()                                
                            merged_question = merged_question.replace("### Merged Question", "")
                            judge_input = MULTI_HOP_JUDGE + f"""\nMerged Query: {merged_question}\nSeries of question and answers:Q1: {Q1}\nA1: {A1}\nQ2: {Q2}"""
                            judge_response = self.chat_rits_mistral.invoke(judge_input)
                            if type(judge_response)==AIMessage:
                                judge_response = judge_response.model_dump()["content"]
                            reasoning = judge_response.split("<reasoning>")[-1].split("</reasoning>")[0].strip()
                            final_verdict = judge_response.split("<final_verdict>")[-1].split("</final_verdict>")[0].strip()
                            rewritten_question = ""
                            if final_verdict == "mismatch":
                                rewritten_question = judge_response.split("<new_query>")[-1].split("</new_query>")[0].strip()
                            gold_sequence_api = self.data[path[start_index + 1]]
                            gold_sequence_api["question_type"] = "API"
                            gold_sequence_rag = {"question": rag_questions[self.encode(relation)]["query"], "answer": rag_questions[self.encode(relation)]["answer"], "relation": relation, "rag_doc": rag_questions[self.encode(relation)]["documents"], "question_type": "RAG", "db_id": self.domain}
                            turns.append({
                                "query": merged_question,
                                "answer": self.data[path[start_index + 1]]['answer'],
                                "type": "(RAG-API)",
                                "index": None,
                                "gold_sequence": [gold_sequence_rag, gold_sequence_api],
                                "metadata": {"reasoning": reasoning, "final_verdict": final_verdict, "rewritten_question": rewritten_question}
                            })
                            types += "(RAG-API)"
                            start_index += num_hops
                            hops.append(num_hops)
                        else:
                            gold_sequence_api = self.data[path[start_index + 1]]
                            gold_sequence_api["question_type"] = "API"
                            turns.append({
                                "query": self.data[path[start_index + 1]]['question'],
                                "answer": self.data[path[start_index + 1]]['answer'],
                                "type": "(API)",
                                "gold_sequence": [gold_sequence_api],
                                "index": None
                            })
                            types += "(API)"
                            start_index += num_hops
                            hops.append(1)

                elif num_hops == 3:
                    if start_index + 2 > max_index:
                        continue
                    if isinstance(path[start_index], int):
                        #API_RAG_API
                        if len(self.data[path[start_index]]['answer']) > 2:
                            continue
                        relation = random.choice(path[start_index + 1])
                        if relation:
                            question1 = self.data[path[start_index]]['question']
                            answer1 = self.data[path[start_index]]['answer']
                            question2 = rag_questions[self.encode(relation)]["query"]
                            answer2 = rag_questions[self.encode(relation)]["answer"]
                            question3 = self.data[path[start_index + 2]]['question']
                            answer3 = self.data[path[start_index + 2]]['answer']
                            prompt = MERGE_3_QUESTIONS.format(Q1 = question1, A1 = answer1, Q2 = question2, A2 = answer2, Q3=question3)
                            merged_question = self.chat_rits_mistral.invoke(prompt)
                            if type(merged_question)==AIMessage:
                                merged_question = merged_question.model_dump()["content"]
                            merged_question = merged_question.replace("### Merged Question", "")
                            if "<new_query>" in merged_question and "</new_query>" in merged_question:
                                merged_question = merged_question.split("<new_query>")[-1].split("</new_query>")[0].strip()
                            if "<new_query>" in merged_question:
                                merged_question = merged_question.split("<new_query>")[-1].strip()
                            if "</new_query>" in merged_question:
                                merged_question = merged_question.split("</new_query>")[0].strip()
                            judge_input = MULTI_HOP_JUDGE + f"""\nMerged Query: {merged_question}\nSeries of question and answers:Q1: {question1}\nA1: {answer1}\nQ2: {question2}\nA2: {answer2}\nQ3:  {question3}"""
                            judge_response = self.chat_rits_mistral.invoke(judge_input)
                            if type(judge_response)==AIMessage:
                                judge_response = judge_response.model_dump()["content"]
                            reasoning = judge_response.split("<reasoning>")[-1].split("</reasoning>")[0].strip()
                            final_verdict = judge_response.split("<final_verdict>")[-1].split("</final_verdict>")[0].strip()
                            rewritten_question = ""
                            if final_verdict == "mismatch":
                                rewritten_question = judge_response.split("<new_query>")[-1].split("</new_query>")[0].strip()
                            gold_sequence_api1 = self.data[path[start_index]]
                            gold_sequence_api1["question_type"] = "API"
                            gold_sequence_rag = {"question": rag_questions[self.encode(relation)]["query"], "answer": rag_questions[self.encode(relation)]["answer"], "relation": relation, "rag_doc": rag_questions[self.encode(relation)]["documents"], "question_type": "RAG", "db_id": self.domain}
                            gold_sequence_api2 = self.data[path[start_index + 2]]
                            gold_sequence_api2["question_type"] = "API"
                            turns.append({
                                "query": merged_question,
                                "answer": answer3,
                                "type": "(API-RAG-API)",
                                "index": None,
                                "gold_sequence": [gold_sequence_api1, gold_sequence_rag, gold_sequence_api2],
                                "metadata": {"reasoning": reasoning, "final_verdict": final_verdict, "rewritten_question": rewritten_question}
                            })
                            types += "(API-RAG-API)"
                            # gold_sequence.append([, , ])
                            start_index += num_hops
                            hops.append(3)
                        else:
                            question1 = self.data[path[start_index]]['question']
                            answer1 = self.data[path[start_index]]['answer']
                            gold_sequence_api1 = self.data[path[start_index]]
                            gold_sequence_api1["question_type"] = "API"
                            turns.append({
                                "query": question1,
                                "answer": answer1,
                                "type": "(API)",
                                "index": None,
                                "gold_sequence": [gold_sequence_api1]
                            })
                            types += "(API)"
                            hops.append(1)


                            question3 = self.data[path[start_index + 2]]['question']
                            answer3 = self.data[path[start_index + 2]]['answer']
                            gold_sequence_api3 = self.data[path[start_index + 2]]
                            gold_sequence_api3["question_type"] = "API"
                            turns.append({
                                "query": question3,
                                "answer": answer3,
                                "type": "(API)",
                                "index": None,
                                "gold_sequence": [gold_sequence_api3]
                            })
                            types += "(API)"
                            hops.append(1)


                            start_index += num_hops
                    
                    else:
                        if len(self.data[path[start_index + 1]]['answer']) > 2:
                            continue
                        relation1 = random.choice(path[start_index])
                        if relation1:
                            question1 = rag_questions[self.encode(relation1)]["query"]
                            answer1 = rag_questions[self.encode(relation1)]["answer"]
                            gold_sequence_rag1 = {"question": question1, "answer": answer1, "relation": relation1, "rag_doc": rag_questions[self.encode(relation1)]["documents"], "question_type": "RAG", "db_id": self.domain}

                        question2 = self.data[path[start_index + 1]]['question']
                        answer2 = self.data[path[start_index + 1]]['answer']
                        gold_sequence_api = self.data[path[start_index + 1]]
                        gold_sequence_api["question_type"] = "API"

                        relation2 = random.choice(path[start_index + 2])
                        if relation2:
                            question3 = rag_questions[self.encode(relation2)]["query"]
                            answer3 = rag_questions[self.encode(relation2)]["answer"]
                            gold_sequence_rag3 = {"question": question3, "answer": answer3, "relation": relation2, "rag_doc": rag_questions[self.encode(relation2)]["documents"], "question_type": "RAG", "db_id": self.domain}

                        if relation1 and relation2:
                            prompt = MERGE_3_QUESTIONS.format(Q1 = question1, A1 = answer1, Q2 = question2, A2 = answer2, Q3=question3)
                            merged_question = self.chat_rits_mistral.invoke(prompt)
                            if type(merged_question)==AIMessage:
                                merged_question = merged_question.model_dump()["content"]
                            merged_question = merged_question.replace("### Merged Question", "")
                            if "<new_query>" in merged_question and "</new_query>" in merged_question:
                                merged_question = merged_question.split("<new_query>")[-1].split("</new_query>")[0].strip()
                            if "<new_query>" in merged_question:
                                merged_question = merged_question.split("<new_query>")[-1].strip()
                            if "</new_query>" in merged_question:
                                merged_question = merged_question.split("</new_query>")[0].strip()
                            judge_input = MULTI_HOP_JUDGE + f"""\nMerged Query: {merged_question}\nSeries of question and answers:Q1: {question1}\nA1: {answer1}\nQ2: {question2}\nA2: {answer2}\nQ3:  {question3}"""
                            judge_response = self.chat_rits_mistral.invoke(judge_input)
                            if type(judge_response)==AIMessage:
                                judge_response = judge_response.model_dump()["content"]
                            reasoning = judge_response.split("<reasoning>")[-1].split("</reasoning>")[0].strip()
                            final_verdict = judge_response.split("<final_verdict>")[-1].split("</final_verdict>")[0].strip()
                            rewritten_question = ""
                            if final_verdict == "mismatch":
                                rewritten_question = judge_response.split("<new_query>")[-1].split("</new_query>")[0].strip()
                            turns.append({
                                "query": merged_question,
                                "answer": relation2[2],
                                "type": "(RAG-API-RAG)",
                                "gold_sequence": [gold_sequence_rag1,
                                                gold_sequence_api,
                                                gold_sequence_rag3],
                                "index": None,
                                "metadata": {"reasoning": reasoning, "final_verdict": final_verdict, "rewritten_question": rewritten_question}
                            })
                            types += "(RAG-API-RAG)"                
                            start_index += num_hops
                            hops.append(3)
                        elif relation1 and not relation2:
                            #RAG_BEFORE_API
                            prompt = MERGE_2_QUESTIONS.format(Q1=question1, A1 = answer1, Q2=question2)
                            merged_question = self.chat_rits_mistral.invoke(prompt)
                            if type(merged_question)==AIMessage:
                                merged_question = merged_question.model_dump()["content"]
                            merged_question = merged_question.replace("### Merged Question", "")
                            if "<new_query>" in merged_question and "</new_query>" in merged_question:
                                merged_question = merged_question.split("<new_query>")[-1].split("</new_query>")[0].strip()
                            if "<new_query>" in merged_question:
                                merged_question = merged_question.split("<new_query>")[-1].strip()
                            if "</new_query>" in merged_question:
                                merged_question = merged_question.split("</new_query>")[0].strip()
                            judge_input = MULTI_HOP_JUDGE + f"""\nMerged Query: {merged_question}\nSeries of question and answers:Q1: {question1}\nA1: {answer1}\nQ2: {question2}"""
                            judge_response = self.chat_rits_mistral.invoke(judge_input)
                            if type(judge_response)==AIMessage:
                                judge_response = judge_response.model_dump()["content"]
                            reasoning = judge_response.split("<reasoning>")[-1].split("</reasoning>")[0].strip()
                            final_verdict = judge_response.split("<final_verdict>")[-1].split("</final_verdict>")[0].strip()
                            rewritten_question = ""
                            if final_verdict == "mismatch":
                                rewritten_question = judge_response.split("<new_query>")[-1].split("</new_query>")[0].strip()
                            turns.append({
                                "query": merged_question,
                                "answer": answer2,
                                "type": "(RAG-API)",
                                "index": None,
                                "gold_sequence": [gold_sequence_rag1, gold_sequence_api],
                                "metadata": {"reasoning": reasoning, "final_verdict": final_verdict, "rewritten_question": rewritten_question}
                            })
                            types += "(RAG-API)"
                            start_index += num_hops
                            hops.append(2)

                        elif relation2 and not relation1:
                            #API_BEFORE_RAG
                            prompt = MERGE_2_QUESTIONS.format(Q1=question2, A1 = answer2, Q2 = question3)
                            merged_question = self.chat_rits_mistral.invoke(prompt)
                            if type(merged_question)==AIMessage:
                                merged_question = merged_question.model_dump()["content"]
                            merged_question = merged_question.replace("### Merged Question", "")
                            if "<new_query>" in merged_question and "</new_query>" in merged_question:
                                merged_question = merged_question.split("<new_query>")[-1].split("</new_query>")[0].strip()
                            if "<new_query>" in merged_question:
                                merged_question = merged_question.split("<new_query>")[-1].strip()
                            if "</new_query>" in merged_question:
                                merged_question = merged_question.split("</new_query>")[0].strip()
                            judge_input = MULTI_HOP_JUDGE + f"""\nMerged Query: {merged_question}\nSeries of question and answers:Q1: {question2}\nA1: {answer2}\nQ2: {question3}"""
                            judge_response = self.chat_rits_mistral.invoke(judge_input)
                            if type(judge_response)==AIMessage:
                                judge_response = judge_response.model_dump()["content"]
                            reasoning = judge_response.split("<reasoning>")[-1].split("</reasoning>")[0].strip()
                            final_verdict = judge_response.split("<final_verdict>")[-1].split("</final_verdict>")[0].strip()
                            rewritten_question = ""
                            if final_verdict == "mismatch":
                                rewritten_question = judge_response.split("<new_query>")[-1].split("</new_query>")[0].strip()
                            turns.append({
                                "query": merged_question,
                                "answer": answer3,
                                "type": "(API-RAG)",
                                "gold_sequence": [gold_sequence_api, gold_sequence_rag3],
                                "index": None,
                                "metadata": {"reasoning": reasoning, "final_verdict": final_verdict, "rewritten_question": rewritten_question}
                            })
                            types += "(API-RAG)"
                            start_index += num_hops
                            hops.append(2)
                        else:
                            #API
                            turns.append({
                            "query": self.data[path[start_index + 1]]['question'],
                            "answer": self.data[path[start_index + 1]]['answer'],
                            "type": "(API)",
                            "gold_sequence": [gold_sequence_api],
                            "index": None
                        })
                            types += "(API)"
                            start_index += num_hops
                            hops.append(1)


            sample["turns"] = turns
            sample["num_turns"] = len(turns)
            sample["num_hops"] = hops
            sample["type"] = types
            multi_turn.append(sample)

        return multi_turn
    
    def improve_multi_turn(self, multi_turn_incorporated):
        output_data = []
        sample_id = 0
        for idx, data in tqdm(enumerate(multi_turn_incorporated), total=len(multi_turn_incorporated), desc="Turning into multi-turn format!"):
            if data["num_turns"] == 1 and data["type"] == "(API)":
                continue
            if data["num_turns"] > MAX_NUM_TURNS:
                i = 0
                while i < data["num_turns"]:
                    curr_turns = min(random.randint(2, 7), data["num_turns"] - i)
                    turns = data["turns"][i: min(i + curr_turns, data["num_turns"])]
                    prev_question = turns[0]["query"]
                    prev_answer = turns[0]["answer"]
                    for turn_id, element in enumerate(turns):
                        input_data = []
                        if turn_id == 0 or (isinstance(prev_answer, list) and len(prev_answer) > 1):
                            continue
                        input_data.append({"query": prev_question, "answer": prev_answer})
                        input_data.append({"query": element["query"]})
                        prompt = ENHANCE_MULTI_TURN.format(answer = prev_answer, input = str(input_data), example = ENHANCE_MULTI_TURN_EXAMPLE)
                        enhanced_conversation = self.chat_rits_mistral.invoke(prompt)
                        if type(enhanced_conversation)==AIMessage:
                            enhanced_conversation = enhanced_conversation.model_dump()["content"]
                        new_query = enhanced_conversation
                        if "<new_query>" in new_query and "</new_query>" in new_query:
                            new_query = new_query.split("<new_query>")[-1].split("</new_query>")[0].strip()
                        if "<new_query>" in new_query:
                            new_query = new_query.split("<new_query>")[-1].strip()
                        if "</new_query>" in new_query:
                            new_query = new_query.split("</new_query>")[0].strip()
                        prev_question = element["query"]
                        prev_answer = element["answer"]
                        element["query"] = new_query
                        element["original"] = prev_question
                        try:
                            element["decision"] = enhanced_conversation.split("<understand>")[-1].split("</understand>")[0].strip()
                        except:
                            element["decision"] = None
                    types = ''.join([dictionary["type"] for dictionary in turns])
                    output_data.append({"sample_id": sample_id, "dataset_name": data["dataset_name"], "turns": turns,"type": types, "num_turns": len(turns), "num_hops": data["num_hops"][i: min(i + curr_turns, data["num_turns"])], "tools": None})
                    sample_id += 1
                    i += curr_turns

            else:
                prev_question = data["turns"][0]["query"]
                prev_answer = data["turns"][0]["answer"]
                for turn_id, element in enumerate(data["turns"]):
                    input_data = []
                    if turn_id == 0 or (isinstance(prev_answer, list) and len(prev_answer) > 1):
                        continue
                    input_data.append({"query": prev_question, "answer": prev_answer})
                    input_data.append({"query": element["query"]})
                    prompt = ENHANCE_MULTI_TURN.format(answer = prev_answer, input = str(input_data), example = ENHANCE_MULTI_TURN_EXAMPLE)
                    enhanced_conversation = self.chat_rits_mistral.invoke(prompt)
                    if type(enhanced_conversation)==AIMessage:
                        enhanced_conversation = enhanced_conversation.model_dump()["content"]
                    new_query = enhanced_conversation
                    if "<new_query>" in new_query and "</new_query>" in new_query:
                        new_query = new_query.split("<new_query>")[-1].split("</new_query>")[0].strip()
                    if "<new_query>" in new_query:
                        new_query = new_query.split("<new_query>")[-1].strip()
                    if "</new_query>" in new_query:
                        new_query = new_query.split("</new_query>")[0].strip()   
                    prev_question = element["query"]
                    prev_answer = element["answer"]
                    element["query"] = new_query
                    element["original"] = prev_question
                    try:
                        element["decision"] = enhanced_conversation.split("<understand>")[-1].split("</understand>")[0].strip()
                    except:
                        element["decision"] = None
                data["sample_id"] = sample_id
                sample_id += 1
                output_data.append(data)
        return output_data





        

                






            
        
        
        

