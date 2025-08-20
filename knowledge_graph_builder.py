import json
import os
import json
import sqlite3
import re
from pathlib import Path
from bs4 import BeautifulSoup
import itertools
import time
import string
import requests
from sqlglot import parse_one, exp
import networkx as nx
from pyvis.network import Network
from pathlib import Path
from tqdm import tqdm
import dateparser
from thefuzz import fuzz
import spacy
from typing import List, Tuple
from prompt import SYSTEM_PROMPT_MISTRAL
from config import mistral_config
import string
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from langchain_core.messages.ai import AIMessage

nlp = spacy.load("en_core_web_sm")
translator = str.maketrans('', '', string.punctuation)


class SQLAnswerAndParameterExtractor:
    def __init__(self, db_folder, domain, database, json_file):
        self.db_folder = db_folder
        self.domain = domain
        self.database = database
        self.json_file = json_file
        self.sqlite_path = os.path.join(db_folder, domain, f"{domain}.sqlite")
        self.output_json = os.path.join(f'output/{database}/{domain}', f"{domain}.json")
        os.makedirs(f'output/{database}/{domain}', exist_ok=True)
    
    def extract_parameters(self, sql):
        results = []
        try:
            parsed = parse_one(sql, read="mysql")
            for literal in parsed.find_all(exp.Literal):
                if literal.is_string:
                    value = literal.this
                    parent = literal.parent
                    if isinstance(parent, exp.Like):
                        core = value.strip('%')
                        if len(core) > 1 and any(char.isalnum() for char in core):
                            results.append(core)
                    else:
                        value = value.strip('%')
                        if len(value) > 1 and any(char.isalnum() for char in value):
                            results.append(value)
        except:
            pass
        return sorted(set(results))
    
    def extract_answers_and_parameters(self):
        with open(self.json_file) as f:
            data = json.load(f)
        data = [item for item in data if item.get('db_id') == self.domain]
        conn = sqlite3.connect(self.sqlite_path)
        cursor = conn.cursor()

        for item in tqdm(data):
            sql_key = 'SQL' if self.database == 'bird' else 'query'
            if sql_key in item:
                try:
                    cursor.execute(item[sql_key])
                    item['answer'] = list(set(cursor.fetchall()))
                except:
                    item['answer'] = []
                if item['answer'] is not None:
                    try:
                        item['answer'] = sorted(item['answer'], key=lambda x: [(item is None, item) for item in x])
                    except: pass
                
                item['parameters'] = self.extract_parameters(item[sql_key])

        conn.close()
        with open(self.output_json, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Answers and parameters extracted to {self.output_json}")

class KnowledgeGraphBuilder:
    def __init__(self, base_directory: str, domain: str, entity_directory: str):
        self.base_directory = Path(base_directory)
        self.domain = domain
        self.base_path = self.base_directory / domain
        self.paths = self._configure_paths()
        self.translator = str.maketrans('', '', string.punctuation)
        self.nlp = spacy.load("en_core_web_sm")

        self.entities = self.load_entity_names(entity_directory)
        self.label_cache = self._load_cache(self.paths['label_cache'])
        self.qid_cache = self._load_cache(self.paths['qid_cache'])
    
    def _normalize_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", "", text)
        return text.strip()
    
    def load_entity_names(self, file_path: str) -> List[Tuple[str, List[str]]]:
        entities = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 2:
                    continue
                wikiid = parts[0]
                names = [self._normalize_text(n) for group in parts[1:] for n in group.split(',')]
                entities.append((wikiid, names))
        return entities

    def match_entity(self, input_entity: str) -> str:
        input_norm = self._normalize_text(input_entity)
        for wikiid, names in self.entities:
            if input_norm in names:
                return wikiid
        return None
    
    def _configure_paths(self):
        self.base_path.mkdir(parents=True, exist_ok=True)
        return {
            'data': self.base_path / f"{self.domain}.json",
            'qid_cache': self.base_path / 'wikidata_qid_cache.json',
            'label_cache': self.base_path / 'wikidata_label_cache.json',
            'output_html': self.base_path / 'knowledge_graph.html'
        }

    def _load_cache(self, path):
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return {}

    def _save_cache(self, cache, path):
        with open(path, 'w') as f:
            json.dump(cache, f)

    def _flatten(self, nested_list):
        result = []
        for item in nested_list:
            if isinstance(item, list):
                result.extend(self._flatten(item))
            elif isinstance(item, str):
                result.extend([s.strip() for s in item.split(';') if s.strip()])
            else:
                result.append(item)
        return result

    def _is_currency(self, s):
        pattern = r'^\$[\d, ]+(\.\d+)?\$?$'
        return bool(re.match(pattern, s.strip()))

    def _is_number(self, s):
        try:
            float(s.replace(',', ''))
            return True
        except ValueError:
            return False

    def _is_date(self, s):
        return dateparser.parse(s, settings={'STRICT_PARSING': True}) is not None

    def _is_number_only(self, s):
        s_clean = ''.join(ch for ch in s if ch not in string.punctuation)
        return s_clean.isdigit()

    def _classify_string(self, s):
        try:
            if self._is_currency(s):
                return 'currency'
            elif self._is_number(s) or self._is_number_only(s):
                return 'number'
            elif self._is_date(s):
                return 'date'
            else:
                return 'string'
        except:
            return 'string'

    def _get_wikidata_label(self, qid, delay=1):
        if qid in self.qid_cache:
            return self.qid_cache[qid]
        try:
            response = requests.get(
                f'https://www.wikidata.org/wiki/Special:EntityData/{qid}.json',
                timeout=10
            )
            if response.status_code == 200:
                entity = response.json()['entities'].get(qid, {})
                label = entity.get('labels', {}).get('en', {}).get('value', qid)
                self.qid_cache[qid] = label
                self._save_cache(self.qid_cache, self.paths['qid_cache'])
                return label
        except Exception:
            pass
        time.sleep(delay)

    def _get_wikipedia_page(self, title):
        search_url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": title,
            "format": "json",
        }
        try:
            response = requests.get(search_url, params=params, timeout=10)
        except requests.exceptions.RequestException as e:
            return None
        data = response.json()
        search_results = data.get("query", {}).get("search", [])
        if search_results:
            page_title = search_results[0]["title"].replace(" ", "_")
            return f"https://en.wikipedia.org/wiki/{page_title}"
        return None

    def _get_wikidata_id(self, label):
        if label in self.label_cache:
            return self.label_cache[label]
        if len(label) <= 1:
            self.label_cache[label] = None
            self._save_cache(self.label_cache, self.paths['label_cache'])
        classified_string = self._classify_string(label)
        if classified_string in ["currency", "number"]:
            self.label_cache[label] = None
            self._save_cache(self.label_cache, self.paths['label_cache'])
            return None

        search_for = f"{label} {self.domain}" if classified_string == "string" else label
        try:
            wiki_url = self._get_wikipedia_page(search_for)  
            page_title = wiki_url.split("/wiki/")[-1]
            api_url = "https://en.wikipedia.org/w/api.php"
            params = {
                "action": "query",
                "prop": "pageprops",
                "titles": page_title,
                "format": "json"
            }
            response = requests.get(api_url, params=params, timeout=10)
            data = response.json()
            pages = data.get("query", {}).get("pages", {})
            
            label_lower = label.lower().translate(self.translator)
            for page in pages.values():
                page_title_lower = page["title"].lower().translate(self.translator)
                qid = page.get("pageprops", {}).get("wikibase_item")
                if ((label_lower in page_title_lower or page_title_lower in label_lower) or 
                    (fuzz.token_set_ratio(label_lower, page_title_lower) >= 80)) and qid:
                    self.label_cache[label] = qid
                    self._save_cache(self.label_cache, self.paths['label_cache'])
                    return qid
        except:
            qid = self.match_entity(label)
            self.label_cache[label] = qid
            self._save_cache(self.label_cache, self.paths['label_cache'])
            return qid

        self.label_cache[label] = None
        self._save_cache(self.label_cache, self.paths['label_cache'])
        return None

    def build_graph(self):
        with open(self.paths['data']) as f:
            data = json.load(f)

        entities = {'answer': set(), 'parameter': set()}
        pbar = tqdm(data, desc="Processing data")
        for entry in pbar:
            raw_answers = self._flatten(entry.get('answer', []))
            answers = [a for a in (self._get_wikidata_id(str(item)) for item in raw_answers) if a]
            entities['answer'].update(answers)

            raw_params = self._flatten(entry.get('parameters', []))
            params = [p for p in (self._get_wikidata_id(str(item)) for item in raw_params) if p]
            entities['parameter'].update(params)

        self.qid_cache = {value: key for key, value in self.label_cache.items()}
        self._save_cache(self.qid_cache, self.paths['qid_cache'])

        # Load relationships
        relationships = []
        for split in ['train', 'valid', 'test']:
            rel_file = f"../wikidata5m_transductive/wikidata5m_transductive_{split}.txt"
            with open(rel_file) as f:
                relationships += [line.strip().split('\t') for line in f if len(line.split('\t')) == 3]

        G = nx.DiGraph()
        common_entities = entities['answer'] & entities['parameter']

        for qid in entities['answer'] | entities['parameter']:
            if qid in common_entities:
                color = 'black'
                node_type = 'both'
            elif qid in entities['answer']:
                color = '#87CEEB'
                node_type = 'answer'
            else:
                color = '#98FB98'
                node_type = 'parameter'
            label = self.qid_cache.get(qid, qid)
            G.add_node(qid, type=node_type, color=color, title=label)

        for subj, rel, obj in tqdm(relationships, desc="Building graph"):
            if subj in self.qid_cache and obj in self.qid_cache:
                subj_type = G.nodes.get(subj, {}).get('type')
                obj_type = G.nodes.get(obj, {}).get('type')
                rel_label = self._get_wikidata_label(rel)
                if subj_type in ['answer', 'both'] and obj_type in ['parameter', 'both']:
                    G.add_edge(subj, obj, label=rel_label, title=rel)
                elif subj_type in ['parameter', 'both'] and obj_type in ['answer', 'both']:
                    G.add_edge(obj, subj, label=rel_label, title=rel)

        net = Network(height='800px', width='100%', directed=True)
        net.from_nx(G)
        net.show_buttons(filter_=['physics'])
        net.show(str(self.paths['output_html']), notebook=False)
        print(f"Interactive visualization saved to {self.paths['output_html']}")

        return G
    
class GraphRelationshipAugmenter:
    def __init__(self, domain, folder, db_folder, chat_rits_mistral):
        self.domain = domain
        self.folder = Path(folder)
        self.domain_json_path = self.folder / f"{self.domain}.json"
        self.is_rag_present_in_database = self.folder /f"is_rag_present_in_database.json"
        self.html_path = self.folder / "knowledge_graph.html"
        self.id_to_title_path = self.folder / "wikidata_qid_cache.json"
        self.title_to_id_path = self.folder / "wikidata_label_cache.json"
        self.db_folder = db_folder
        self.sqlite_path = os.path.join(db_folder, domain, f"{domain}.sqlite")
        self.chat_rits_mistral = chat_rits_mistral

    @staticmethod
    def extract_vis_dataset(js_code, var_name):
        """
        Extract the array contents from:
        var var_name = new vis.DataSet([...]);
        """
        pattern = rf"{var_name}\s*=\s*new vis\.DataSet\((\[.*?\])\);"
        match = re.search(pattern, js_code, re.DOTALL)
        if not match:
            raise ValueError(f"Could not extract {var_name} from the HTML.")
        json_str = match.group(1)
        return json.loads(json_str)

    def augment_json_with_rag_links(self):
        # Load the JSON data
        domain_database_path = f"{self.db_folder}/{self.domain}/database_description"
        schema = ""
        for filename in os.listdir(domain_database_path):
            if filename.endswith(".csv"):
                file_path = os.path.join(domain_database_path, filename)
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                name = os.path.splitext(filename)[0]  
                schema += f"{name}.csv:\n{content}\n\n" 

        global SYSTEM_PROMPT_MISTRAL
        SYSTEM_PROMPT_MISTRAL = SYSTEM_PROMPT_MISTRAL.replace("{schema}", schema)

        with open(self.domain_json_path, 'r') as f:
            data = json.load(f)

        with open(self.id_to_title_path, 'r') as f:
            id_to_title = json.load(f)

        with open(self.title_to_id_path, 'r') as f:
            title_to_id = json.load(f)

        with open(self.html_path, 'r') as f:
            soup = BeautifulSoup(f, 'html.parser')

        script_tag = soup.find('script', string=lambda x: x and 'var nodes' in x)
        if not script_tag:
            raise ValueError("Could not find network data in HTML file.")

        script_content = script_tag.string
        nodes = self.extract_vis_dataset(script_content, "nodes")
        edges = self.extract_vis_dataset(script_content, "edges")

        # Build outgoing edges dictionary
        outgoing_edges = {}
        for edge in edges:
            from_id = str(edge['from'])
            to_id = str(edge['to'])
            label = edge.get('label', '')
            if from_id not in outgoing_edges:
                outgoing_edges[from_id] = []
            outgoing_edges[from_id].append({
                'relation': label,
                'to_id': to_id,
                'to_title': id_to_title.get(str(to_id), '')
            })

        parameter_to_question_ids = {}
        if os.path.exists(self.is_rag_present_in_database):
            with open(self.is_rag_present_in_database, 'r') as f:
                rag_database = json.load(f)
        else:
            rag_database = {}

        for idx, e in enumerate(data):
            for p in e.get('parameters', []):
                parameter_to_question_ids.setdefault(p, []).append(idx)

        conn = sqlite3.connect(self.sqlite_path)
        cursor = conn.cursor()

        for idx, element in tqdm(enumerate(data), total=len(data), desc="Checking RAG edges!"):
            answer_field = list(itertools.chain.from_iterable(element.get('answer', [])))
            new_dict = {}

            for answer in answer_field:
                matching_node_id = title_to_id.get(answer)
                if matching_node_id is None:
                    continue
                outgoing = outgoing_edges.get(matching_node_id, [])

                if len(outgoing) > 0 and answer not in rag_database:
                    rag_database[answer] = []

                matched_relations = []
                for out_edge in outgoing:
                    question_ids = parameter_to_question_ids.get(id_to_title[out_edge['to_id']])
                    if question_ids is None:
                        continue
                    to_append = [qid for qid in question_ids if qid != idx]
                    relationship = f'"{answer}" "{out_edge["relation"]}" "{id_to_title[out_edge["to_id"]]}"'

                    infoFromDatabase = None
                    for info in rag_database[answer]:
                        try:
                            if relationship == info.get("relationship"):
                                infoFromDatabase = info
                                break
                        except:
                            pass
                                

                    if infoFromDatabase is not None and isinstance(infoFromDatabase, dict):
                        if infoFromDatabase.get("canAnswer"):
                            try:                        
                                cursor.execute(infoFromDatabase["SQL"])
                                result = list(set(cursor.fetchall()))
                                if len(result) > 0:
                                    continue  
                            except:
                                continue

                    else:
                        
                        prompt = SYSTEM_PROMPT_MISTRAL + relationship
                        infoFromDatabase = self.chat_rits_mistral.invoke(prompt)
                        if type(infoFromDatabase)==AIMessage:
                            infoFromDatabase = infoFromDatabase.model_dump()["content"]
                        infoFromDatabase = infoFromDatabase.replace("```json", "").replace("```", "").replace("\\", "\\\\")

                        try:
                            infoFromDatabase = json.loads(infoFromDatabase, strict=False)
                        except:
                            rag_database[answer].append(infoFromDatabase)
                            with open(self.is_rag_present_in_database, 'w') as f:
                                json.dump(rag_database, f, indent=2)
                            continue

                        if infoFromDatabase.get("canAnswer"):
                            try:
                                cursor.execute(infoFromDatabase["SQL"])
                                result = list(set(cursor.fetchall()))
                                infoFromDatabase["result"] = result
                                infoFromDatabase["relationship"] = relationship
                                rag_database[answer].append(infoFromDatabase)
                            except Exception as e:
                                print(e)
                                infoFromDatabase["relationship"] = relationship
                                rag_database[answer].append(infoFromDatabase)
                                with open(self.is_rag_present_in_database, 'w') as f:
                                    json.dump(rag_database, f, indent=2)
                                continue
                        else:
                            infoFromDatabase["relationship"] = relationship
                            rag_database[answer].append(infoFromDatabase)

                    with open(self.is_rag_present_in_database, 'w') as f:
                        json.dump(rag_database, f, indent=2)

                    if len(to_append) > 0:
                        matched_relations.append([
                            out_edge['relation'], id_to_title[out_edge['to_id']], to_append
                        ])

                if matched_relations:
                    new_dict[answer] = matched_relations

            element['links'] = new_dict if new_dict else {}

        with open(self.domain_json_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"JSON file '{self.domain_json_path}' has been updated with 'links' entries.")

    def augment_json_with_direct_links(self):
        with open(self.domain_json_path, 'r') as f:
            data = json.load(f)
        
        parameter_to_question_ids = {}
        for idx, e in enumerate(data):
            for p in e.get('parameters', []):
                parameter_to_question_ids.setdefault(p, []).append(idx)

        for id, element in enumerate(data):
            outgoing_questions = []
            answer_field = list(itertools.chain.from_iterable(element.get('answer', [])))
            for answer in answer_field:
                if answer in parameter_to_question_ids:
                    outgoing_questions.extend(parameter_to_question_ids[answer])
            if id in outgoing_questions:
                outgoing_questions.remove(id)
            element['direct_links'] = list(set(outgoing_questions))

        with open(self.domain_json_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"JSON file '{self.domain_json_path}' has been updated with 'direct links' entries.")

class LinksGraphVizualizer:
    def __init__(self, json_path):
        self.json_path = json_path
        self.graph = {}
        self.data = self.load_json()

    def load_json(self):
        with open(self.json_path, 'r') as f:
            return json.load(f)

    def build_graph(self):
        for element_id, element in enumerate(self.data):
            self.graph[element_id] = []  
            links = element.get("links", {})
            for link_list in links.values():
                for link in link_list:
                    linked_ids = link[2]
                    self.graph[element_id].extend(linked_ids)

    def visualize_graph(self, output_html='links.html'):
        net = Network(notebook=False, directed=True, height="750px", width="100%")
        net.barnes_hut()

        for node_id in self.graph.keys():
            net.add_node(node_id, label=str(node_id), title=f"Node {node_id}")

        for src, targets in self.graph.items():
            for tgt in targets:
                net.add_edge(src, tgt)

        net.show(output_html, notebook=False)

    def run(self, output_html='links.html'):
        self.build_graph()

        self.visualize_graph(output_html)
        print(f"Graph visualization saved to {output_html}")