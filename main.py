import argparse
from knowledge_graph_builder import SQLAnswerAndParameterExtractor, KnowledgeGraphBuilder, GraphRelationshipAugmenter, LinksGraphVizualizer
from final_queries_generation import MultiTurn
import os
import json
from dotenv import load_dotenv
from llm import ChatRits, Langchain_RITS
from langchain_core.messages.ai import AIMessage
from langchain_core.prompts import PromptTemplate
from config import mistral_config
load_dotenv()



def parse_args():
    parser = argparse.ArgumentParser(description='Process SQL queries and create knowledge graph')
    subparsers = parser.add_subparsers(dest='command', required=True)

    extract_parser = subparsers.add_parser('extract_answers', help='Extract answers and parameters from SQL queries')
    extract_parser.add_argument('--json_file', required=True, help='Input (train.json/dev.json) file path')
    extract_parser.add_argument('--db_folder', required=True, help='Database folder path')
    extract_parser.add_argument('--domain', required=True, help='Domain name')
    extract_parser.add_argument('--database', choices=["bird", "spider"], default="bird", help='dataset to transform')
    extract_parser.add_argument("--mode", type=str, default='local', required=True, help="Test Model -- 'local' or 'gb'")

    knowledge_graph_parser = subparsers.add_parser('knowledge_graph', help='Knowledge graph')
    knowledge_graph_parser.add_argument('--base_directory', required=True, help="Path to base directory (output/bird)")
    knowledge_graph_parser.add_argument('--domain', required=True, help='Domain name')
    knowledge_graph_parser.add_argument('--entity_directory', required=True, help="Path to wikidata5m_entity.txt")
    knowledge_graph_parser.add_argument('--match_entities', action="store_true", help="Get the final links only")
    knowledge_graph_parser.add_argument('--db_folder', required=True, help='Database folder path')
    knowledge_graph_parser.add_argument("--mode", type=str, default='local', required=True, help="Test Model -- 'local' or 'gb'")

    vizualize_links = subparsers.add_parser('vizualize_links', help='vizualize the final links obtained')
    vizualize_links.add_argument('--json_file', required=True, help='Input JSON file path (domain.json inside output/bird/<domain>/<domain>.json)')
    vizualize_links.add_argument("--mode", type=str, default='local', required=True, help="Test Model -- 'local' or 'gb'")

    rag_before_api = subparsers.add_parser('rag_before_api', help='Generate RAG before API queries')
    rag_before_api.add_argument('--domain', required=True, help='Domain name')
    rag_before_api.add_argument('--version', required=True, help='Version of dataset generation')
    rag_before_api.add_argument("--mode", type=str, default='local', required=True, help="Test Model -- 'local' or 'gb'")

    api_before_rag = subparsers.add_parser('api_before_rag', help='Generate API before RAG queries')
    api_before_rag.add_argument('--domain', required=True, help='Domain name')
    api_before_rag.add_argument('--version', required=True, help='Version of dataset generation')
    api_before_rag.add_argument("--mode", type=str, default='local', required=True, help="Test Model -- 'local' or 'gb'")

    multi_turn = subparsers.add_parser('multi_turn', help='Generate multi turn queries')
    multi_turn.add_argument('--domain', required=True, help='Domain name')
    multi_turn.add_argument('--version', required=True, help='Version of dataset generation')
    multi_turn.add_argument("--mode", type=str, default='local', required=True, help="Test Model -- 'local' or 'gb'")
    multi_turn.add_argument("--input_dir", type=str, required=True, default = "output", help="Input")
    multi_turn.add_argument("--output_dir", type=str, required=True, default = "output/bird", help="Output")

    return parser.parse_args()

def main():
    args = parse_args()
    chat_rits_mistral = ChatRits(mistral_config[args.mode])        

    if args.command == 'extract_answers':
        extractor = SQLAnswerAndParameterExtractor(
            db_folder=args.db_folder,
            domain=args.domain,
            database=args.database,
            json_file=args.json_file
        )
        extractor.extract_answers_and_parameters()
    
    elif args.command == "knowledge_graph":
        directory = f"{args.base_directory}/{args.domain}"
        if args.match_entities:
            augmenter = GraphRelationshipAugmenter(args.domain, directory, args.db_folder, chat_rits_mistral)
            augmenter.augment_json_with_rag_links()
            augmenter.augment_json_with_direct_links()
        else:
            builder = KnowledgeGraphBuilder(args.base_directory, args.domain, args.entity_directory)
            graph = builder.build_graph()
            print("Graph has", len(graph.nodes()), "nodes and", len(graph.edges()), "edges.")
            augmenter = GraphRelationshipAugmenter(args.domain, directory, args.db_folder, chat_rits_mistral)
            augmenter.augment_json_with_rag_links()
            augmenter.augment_json_with_direct_links()

    elif args.command == "vizualize_links":
        visualizer = LinksGraphVizualizer(args.json_file)
        output_path = args.json_file[:(args.json_file.rfind('/'))] + "/links.html"
        visualizer.run(output_path)  
    elif args.command == "api_before_rag":
        domain = args.domain
        input_file = f'./output/bird/{domain}/{domain}.json'
        label_cache_file = f'./output/bird/{domain}/wikidata_label_cache.json'
        multi_turn = MultiTurn(domain, input_file, label_cache_file)
        relationships = f'./output/bird/{domain}/relationships.json'
        if os.path.exists(relationships):
            with open(relationships, 'r') as f:
                club_relationships = json.load(f)
        else:
            club_relationships = multi_turn.club_rag_e1_e2()
            with open(relationships, 'w') as file:
                json.dump(club_relationships, file)
        rag_questions_path = f'./output/bird/{domain}/rag_questions_{args.version}.json'
        if os.path.exists(rag_questions_path):
            with open(rag_questions_path, 'r') as f:
                rag_questions = json.load(f)
        else:
            rag_questions = multi_turn.create_rag_only_questions(club_relationships)
            with open(rag_questions_path, 'w') as f:
                json.dump(rag_questions, f)
        api_before_rag_incorporated = multi_turn.create_api_before_rag_questions(rag_questions)
        with open(f'./output/bird/{domain}/api_before_rag_{args.version}.json', 'w') as file:
            json.dump(api_before_rag_incorporated, file)

    elif args.command == "rag_before_api":
        domain = args.domain
        input_file = f'./output/bird/{domain}/{domain}.json'
        label_cache_file = f'./output/bird/{domain}/wikidata_label_cache.json'
        multi_turn = MultiTurn(domain, input_file, label_cache_file)
        relationships = f'./output/bird/{domain}/relationships.json'
        if os.path.exists(relationships):
            with open(relationships, 'r') as f:
                club_relationships = json.load(f)
        else:
            club_relationships = multi_turn.club_rag_e1_e2()
            with open(relationships, 'w') as file:
                json.dump(club_relationships, file)
        rag_questions_path = f'./output/bird/{domain}/rag_questions_{args.version}.json'
        if os.path.exists(rag_questions_path):
            with open(rag_questions_path, 'r') as f:
                rag_questions = json.load(f)
        else:
            rag_questions = multi_turn.create_rag_only_questions(club_relationships)
            with open(rag_questions_path, 'w') as f:
                json.dump(rag_questions, f)
        rag_before_api_incorporated = multi_turn.create_rag_before_api_questions(rag_questions)
        with open(f'./output/bird/{domain}/rag_before_api_{args.version}.json', 'w') as file:
            json.dump(rag_before_api_incorporated, file)
        
    elif args.command == "multi_turn":
        domain = args.domain
        os.makedirs(os.path.join(args.output_dir, domain), exist_ok=True)
        input_dir = os.path.join(args.input_dir, "bird")
        input_file = os.path.join(input_dir, domain, f"{domain}.json")
        label_cache_file = os.path.join(input_dir, domain, 'wikidata_label_cache.json')
        multi_turn = MultiTurn(domain, input_file, label_cache_file, chat_rits_mistral)
        relationships = os.path.join(input_dir, domain, 'relationships.json')
        if os.path.exists(relationships):
            with open(relationships, 'r') as f:
                club_relationships = json.load(f)
        else:
            club_relationships = multi_turn.club_rag_e1_e2()
            # with open(relationships, 'w') as file:
            #     json.dump(club_relationships, file)
        rag_questions_path = os.path.join(input_dir, domain, f'rag_questions_{args.version}.json')
        if os.path.exists(rag_questions_path):
            with open(rag_questions_path, 'r') as f:
                rag_questions = json.load(f)
        else:
            rag_questions = multi_turn.create_rag_only_questions(club_relationships)
            # with open(rag_questions_path, 'w') as f:
            #     json.dump(rag_questions, f)
        multi_turn_intial_path = os.path.join(args.output_dir, domain, f'multi_turn_initial_{args.version}.json')
        if os.path.exists(os.path.join(input_dir, domain, f'multi_turn_initial_{args.version}.json')):
            with open(os.path.join(input_dir, domain, f'multi_turn_initial_{args.version}.json')) as f:
                multi_turn_incorporated = json.load(f)
        else:
            multi_turn_incorporated = multi_turn.run(rag_questions)
            with open(multi_turn_intial_path, 'w') as file:
                json.dump(multi_turn_incorporated, file)
            print("Built original!")
            task_jsonl={
                "databuilder_spec": {
                    "name": f"multi_turn_{domain}",
                    "blocks": [
                    {
                        "base_url":  mistral_config[args.mode]["end_point"],
                        "model_id_or_path": mistral_config[args.mode]["model_name"]
                    }
                    ],
                    "metadata": {
                    "version": 4
                    }
                },
                }
            with open(os.path.join(args.output_dir, domain, "task_card.jsonl"), "w", encoding="UTF-8") as task_writer:
                task_writer.write(json.dumps(task_jsonl) + "\n")
            print("Task JSONL file created.")

        multi_turn_incorporated = multi_turn.improve_multi_turn(multi_turn_incorporated)
        with open(os.path.join(args.output_dir, domain, f'multi_turn_{args.version}.json'), 'w') as file:
            json.dump(multi_turn_incorporated, file)
        task_jsonl={
                "databuilder_spec": {
                    "name": f"multi_turn_{domain}_incorporated",
                    "blocks": [
                    {
                        "base_url":  mistral_config[args.mode]["end_point"],
                        "model_id_or_path": mistral_config[args.mode]["model_name"]
                    }
                    ],
                    "metadata": {
                    "version": 1
                    }
                },
                }
        with open(os.path.join(args.output_dir, domain, "task_card.jsonl"), "w", encoding="UTF-8") as task_writer:
            task_writer.write(json.dumps(task_jsonl) + "\n")
        print("Task JSONL file created.")



if __name__ == "__main__":
    main()