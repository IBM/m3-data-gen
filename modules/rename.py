import os

root_dir = '/proj/m3benchmark/raavi/multi-turn-multi-hop/output_v2/bird'  

for dirpath, dirnames, filenames in os.walk(root_dir):
    for filename in filenames:
        if filename == 'rag_questions_v2.json':
            old_path = os.path.join(dirpath, filename)
            new_path = os.path.join(dirpath, 'rag_questions_v4.json')
            os.rename(old_path, new_path)
            print(f'Renamed: {old_path} -> {new_path}')
