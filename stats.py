import os
import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import seaborn as sns
from collections import defaultdict, Counter
import pandas as pd
from tqdm import tqdm

input_json = "/proj/m3benchmark/raavi/api-sequence-test/balanced/test_v2/unseen/unseen_v7.json"
plot_dir = "/proj/m3benchmark/raavi/api-sequence-test/balanced/test_v2/unseen/plots_v7"
version = "v7" 

def add_question(q_type_raw, domain, q, question_sets, question_domain_sets):
    if not q:
        return
    q_type = q_type_raw.strip("()")  # Remove surrounding parentheses
    if q_type not in question_sets or q_type not in question_domain_sets:
        return  # Skip if q_type isn't recognized

    if domain not in question_domain_sets[q_type]:
        question_domain_sets[q_type][domain] = set()
    question_domain_sets[q_type][domain].add(q)
    question_sets[q_type].add(q)



def plot_question_domain_pies(question_domain_sets, output_dir):
    os.makedirs(output_dir, exist_ok=True)    
    all_domains = sorted({
        inner_key
        for inner_dict in question_domain_sets.values() 
        for inner_key in inner_dict.keys()
    })
    
    cmap = cm.get_cmap('tab20', len(all_domains))  # tab20 or any qualitative colormap
    domain_to_color = {domain: mcolors.to_hex(cmap(i)) for i, domain in enumerate(all_domains)}

    for outer_key, inner_dict in question_domain_sets.items():
        # Get non-empty items and sort them
        count_items = [(k, len(v)) for k, v in inner_dict.items() if len(v) > 0]
        print(outer_key)
        print(inner_dict.keys())
        if not count_items:
            continue

        # Sort by count descending
        count_items.sort(key=lambda x: x[1], reverse=True)

        # Separate top 10 and rest
        top_items = count_items[:10]
        other_items = count_items[10:]

        # Build sizes and labels
        sizes = [v for _, v in top_items + other_items]
        labels = [k for k, _ in top_items] + [''] * len(other_items)

        # Get colors in the same order
        colors = [domain_to_color[k] for k, _ in top_items] + \
                 [domain_to_color[k] for k, _ in other_items]

        # Custom autopct: only for top 10
        def autopct_func(pct):
            total = sum(sizes)
            absolute = int(round(pct * total / 100.0))
            idx = sizes.index(absolute) if absolute in sizes else -1
            return f'{pct:.1f}%' if idx > -1 and idx < 10 else ''

        # Plot
        plt.figure(figsize=(6, 6))
        wedges, texts, autotexts = plt.pie(
            sizes,
            labels=labels,
            autopct=autopct_func,
            startangle=140,
            colors=colors,
            textprops={'fontsize': 8}
        )

        # Remove unwanted percentage texts beyond top 10
        for i, text in enumerate(autotexts):
            if i >= 10:
                text.set_text('')

        plt.title(f'Pie Chart for {outer_key}', fontsize=10)
        plt.axis('equal')

        output_path = os.path.join(output_dir, f"{outer_key}_pie_chart_top_10.png")
        plt.savefig(output_path, dpi=150)
        plt.close()

sns.set(style="whitegrid", palette="muted", font_scale=1.2)


os.makedirs(plot_dir, exist_ok=True)

result = defaultdict(lambda: {i: 0 for i in range(1, 8)})
domain_level_stats = {}
# === 1. Question Type Counts ===
question_sets = {
    "API": set(),
    "RAG": set(),
    "API-RAG": set(),
    "RAG-API": set(),
    "API-RAG-API": set(),
    "RAG-API-RAG": set(),
    "API-API": set(),
    "API-API-API": set(),
    "API-API-RAG": set(),
}

question_domain_sets = {
    "API": {},
    "RAG": {},
    "API-RAG": {},
    "RAG-API": {},
    "API-RAG-API": {},
    "RAG-API-RAG": {},
    "API-API": {},
    "API-API-API": {},
    "API-API-RAG": {},
}

# === 2. Final Table Data ===
table = defaultdict(lambda: defaultdict(int))

# === 3. Domain-level Lengths for Plot ===
domain_lengths = {}
multi_turn_lengths = {}
total_dialogues = 0
total_num_turns = {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "7": 0}
context_response_pairs = 0

data1 = []
with open(input_json) as f:
    data2 = json.load(f)

data = data1 + data2

valid_q_types = ["API-RAG-API", "RAG-API-RAG", "API-RAG", "RAG-API", "API", "RAG", "API-API", "API-API-API", "API-API-RAG"]
for qt in valid_q_types:
    question_sets.setdefault(qt, set())
    question_domain_sets.setdefault(qt, {})
for conv in tqdm(data, total=len(data)):
    num_turns = conv.get("num_turns", 0)
    context_response_pairs += num_turns
    total_num_turns[str(num_turns)] += 1
    domain = conv["dataset_name"]
    if conv["dataset_name"] in domain_level_stats:
        domain_level_stats[conv["dataset_name"]] += 1
    else:
        domain_level_stats[conv["dataset_name"]] = 1
    hop_set = set(conv.get("num_hops", []))
    for hop in hop_set:
        table[hop][num_turns] += 1
    for turn in conv.get("turns", []):
        q = turn.get("query")
        q_type = turn.get("type")
        add_question(q_type, domain, q, question_sets, question_domain_sets)

plot_question_domain_pies(question_domain_sets, plot_dir)
question_type_counts = {k: len(v) for k, v in question_sets.items()}
print("Question Type Counts:")
print(question_type_counts)
distinct_queries = 0
for k, v in question_type_counts.items():
    distinct_queries += v
print(total_num_turns)
file_name = 'total_num_turns_plot.png'
os.makedirs(plot_dir, exist_ok=True)  # Create folder if it doesn't exist
save_path = os.path.join(plot_dir, file_name)

# Create the bar plot
plt.figure(figsize=(8, 5))
plt.bar(total_num_turns.keys(), total_num_turns.values(), color='skyblue')
plt.xlabel('Turn(s)')
plt.ylabel('Number of dialogues')
plt.title('Total Number of dialogues v/s Turn')
plt.tight_layout()

# Save the plot
plt.savefig(save_path)
plt.close()
print("total context response pairs: ", context_response_pairs)
print("Total number of dialogues: ", len(data))

labels = list(domain_level_stats.keys())
sizes = list(domain_level_stats.values())

os.makedirs(plot_dir, exist_ok=True)
output_path = os.path.join(plot_dir, "domain_level_pie_chart.png")

# Create larger figure
fig, ax = plt.subplots(figsize=(8, 8))  # Adjust size as needed

# Create pie chart with small font for labels
wedges, texts, autotexts = ax.pie(
    sizes,
    labels=labels,
    autopct='%1.1f%%',
    startangle=140,
    textprops={'fontsize': 8}  # Smaller font for both labels and percentages
)

# Remove legend
# No legend is added here

# Equal aspect ratio for circular pie
ax.axis('equal')
plt.tight_layout()

# Save the figure
plt.savefig(output_path, dpi=300)

# Optional: display it
plt.show()

for key, value in domain_level_stats.items():
    print(f"{key}: {value}")

sorted_items = sorted(domain_level_stats.items(), key=lambda x: x[1], reverse=True)
top_10 = sorted_items[:10]
other_total = sum(val for _, val in sorted_items[10:])

# Build labels and sizes
labels = [label for label, _ in top_10]
sizes = [value for _, value in top_10]

if other_total > 0:
    labels.append("Other")
    sizes.append(other_total)

# Choose a colormap with enough distinct colors
cmap = plt.get_cmap("tab20")  # 20 distinct colors
colors = [cmap(i) for i in range(len(labels))]

# Ensure output directory exists
os.makedirs(plot_dir, exist_ok=True)
output_path = os.path.join(plot_dir, "domain_level_pie_chart_top10.png")

# Create larger figure
fig, ax = plt.subplots(figsize=(8, 8))

wedges, texts, autotexts = ax.pie(
    sizes,
    labels=labels,
    autopct='%1.1f%%',
    startangle=140,
    colors=colors,              # set distinct colors
    textprops={'fontsize': 8}
)

ax.axis('equal')
plt.tight_layout()
plt.savefig(output_path, dpi=300)
plt.show()

def plot_question_type_pie(question_type_counts, version):
    labels = []
    sizes = []
    for key in question_type_counts:
        labels.append(key)
        sizes.append(question_type_counts[key])

    colors = sns.color_palette("pastel")

    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, colors=colors, startangle=140, autopct='%1.1f%%', textprops={'fontsize': 12})
    plt.title("Distribution of Question Types", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"question_type_pie_{version}.png"))
    # plt.show()

plot_question_type_pie(question_type_counts, version)

# # Final Output:Table of (# dialogs that are x-turn and have y-hop
turn_range = range(1, 8)
hop_levels = [1, 2, 3]
table_rows = []

for hop in hop_levels:
    row = [f"{hop}-hop"]
    for turn in turn_range:
        row.append(table[hop].get(turn, 0))
    table_rows.append(row)


total_row = ["Total"]
for turn in turn_range:
    total_row.append(sum(table[hop].get(turn, 0) for hop in hop_levels))
table_rows.append(total_row)

columns = ["Hop/Turns"] + [f"{i}-turn" for i in turn_range]
table_df = pd.DataFrame(table_rows, columns=columns)
table_df.to_csv(os.path.join(plot_dir, "turns_vs_hops_table.csv"), index=False)



