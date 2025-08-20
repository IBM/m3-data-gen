"""Prompts for multi-turn question generation"""

MERGE_2_QUESTIONS_OLD = """
Merge the two given questions into a single, natural-sounding question that logically combines their information. The merged question should be answerable by {A2}.

Input Format:
Q1: {Q1}
A1: {A1}
Q2: {Q2}
A2: {A2}

Instructions:
- Write a fluent, human-like question that logically fuses the content of Q1 and Q2.
- The answer to the merged question should be {A2}.
- Do not mention {A1} explicitly; instead, refer to it by its role or description given in Q1.
- Preserve key context and relevant entities from Q1.
- Avoid overly long, complex questions. Keep it concise and natural.
- Use contractions and a conversational tone where appropriate.
- Do not introduce any information not present in the input.

Examples:

Q1: Who was the executive producer of Alice in Wonderland?  
A1: Ben Sharpsteen  
Q2: Where was Ben Sharpsteen born?  
A2: Tacoma  
Answer: Where was the executive producer of Alice in Wonderland born?

Q1: Where was Barack Obama born?  
A1: Hawaii  
Q2: In which country is Hawaii located?  
A2: United States  
Answer: Which country is Barack Obama's birthplace located in?

Now, continue with the given input:

Q1: {Q1}  
A1: {A1}  
Q2: {Q2}  
A2: {A2}  
Answer:

"""

MERGE_2_QUESTIONS = """
Now you are assistant to rewrite a query based on first query, response and second query.

The first query, response and second query is given below starting with <context> and end with </context>:
<context>
first query: {Q1}
first query response: {A1}
second query: {Q2}
</context>


In above context, the second query is asking question about {A1}, which rely on getting answer from first query. 
your goal is to create a merged query of first query and second query without mention of first query response.
In merged query, the final answer is the same as second query, but rely on first query and its response to gather necessary information.

Below is the example starting with <example> and end with </example>
<example>
first query: who is the american president during world war II?
first query response: Franklin D. Roosevelt.
second query: when did Franklin D. Roosevelt die?
merged query: when did the american president during world war II die?
</example>

in above example, the answer to merged query is same as second query about the year Franklin D. Roosevelt die, but also rely on first query response to know the american president is Franklin D. Roosevelt specifically. Meanwhile phrase 'Franklin D. Roosevelt' is not in merged query.

Now please generate following below format:
<understand>
read above task description and example, write down what you learned about your task.
</understand>

<new_query>
Write the new query to complete your task based on your understanding. 
Never include {A1} in this new query directly. 
</new_query>
"""

MERGE_3_QUESTIONS = """
Now you are assistant to rewrite a query based on first query, response, second query, response and third query.

The first query, response, second query, response and third query is given below starting with <context> and end with </context>:
<context>
first query: {Q1}
first query response: {A1}
second query: {Q2}
second query response: {A2}
third query: {Q3}
</context>


In above context, the third query is asking question about {A2}, which rely on getting answer from second query.
Similarly, the second query is asking question about {A1}, which rely on getting answer from first query. 
your goal is to create a merged query of first query, second query and third query without mention of first query response and second query response.
In merged query, the final answer is the same as third query, but rely on first query, its response, second query and its response to gather necessary information.

Below is the example starting with <example> and end with </example>
<example>
first query: Which airline operated more flights on 2018/8/1, American Airlines Inc. or Endeavor Air Inc.?
first query response: American Airlines Inc.: AA.
second query: What airport serves as a hub for American Airlines Inc.: AA in New York City?
second query response: New York, NY: John F. Kennedy International
third query: Among the flights on 2018/8/1, how many of them were scheduled to depart from John F. Kennedy International in New York?
merged query: Among the flights on 2018/8/1, how many of them were scheduled to depart from the airport that serves as a hub  for the airline that operated more flights (between American Airlines Inc. and Endeavor Air Inc.) on this day in New York City?
</example>

in above example, the answer to merged query is same as third query about the number of flights scheduled to depart from John F. Kennedy International, but also rely on second query response, second query and the first query response to know that John F. Kennedy International is a hub for American Airlines in New York City and that between American Airlines Inc. and Endeavor Air Inc., American Airlines Inc. operated more flights on 2018/8/1 specifically.
American Airlines Inc.: AA is being included in the merged query because it appeared in the first query. Otherwise it would not have been included since it is an answer to the first query. Endeavor Air Inc. is an important entity that should definitely occur in the merged query.

Now please generate following below format:
<understand>
read above task description and example, write down what you learned about your task. reflect how you would stich the the three questions to remove references about {A1} and {A2} while keeping all other information exactly in-place.
</understand>

<new_query>
Write the new query to complete your task based on your understanding. 
Never include {A1} and {A2} in this new query directly unless they are present in the first query, second query or third query. 
Always include all other entities in the merged query.
Always start with third query, then second query and then first query while forming the merged query without mentioning {A1} and {A2} explicitly.
</new_query>
"""

MERGE_3_QUESTIONS_OLD = """You're given three related questions (Q1, Q2, Q3), each building on the previous one. Your task is to combine them into a single, fluent, and natural-sounding question that incorporates the essential information from all three.

The final merged question must be answerable by {A3}.

Input Format:
Q1: {Q1}
A1: {A1}
Q2: {Q2}
A2: {A2}
Q3: {Q3}
A3: {A3}

Instructions:
- Merge the content of Q1, Q2, and Q3 into one concise question.
- The answer to your merged question must be {A3}.
- Do not include {A1} or {A2} by name; instead, refer to them by their role or description from Q1 and Q2 (e.g., "the person who directed...", "the competitor who...").
- Keep the merged question natural, easy to read, and grammatically fluent.
- Maintain all key context needed to make the merged question logical.
- Use contractions or a conversational tone when appropriate.
- Avoid adding new information or assumptions not already present.

Examples:

Q1: Who is the director of the movie Pinocchio?  
A1: Ben Sharpsteen  
Q2: Ben Sharpsteen is the executive producer of which movie?
A2: Alice in Wonderland  
Q3: Who is the voice actor for the villain of the movie Alice in Wonderland?  
A3: Hans Conried  
Answer: Who is the voice actor for the villain in a movie where the executive producer also directed Pinocchio?

Q1: What is the name of the competitor who has won the most medals?
A1: Michael Fred Phelps, II  
Q2: Which games was Michael Fred Phelps, II a participant in?
A2: 2012 Summer  
Q3: What is the ratio of male to female athletes in the 2012 Summer Olympic?
A3: 1.2587021916630856  
Answer: What is the ratio of male to female athletes at the Games where the competitor who won the most medals participated?

Now, continue with the given input:

Q1: {Q1}  
A1: {A1}  
Q2: {Q2}  
A2: {A2}
Q3: {Q3}
A3: {A3}
Answer:
"""


CREATE_RAG_QUESTION = """
You are given two documents: one about {e1} and one about {e2}.

Task: Generate a single question that includes the entity {e1} and has {e2} as the answer.

- The question must be answerable using the documents about {e1} and {e2}. Only use the information present in documents to create the question.
- Keep the question simple and easy to understand. 
- Keep the question short and include only the essential information needed to answer it from the documents. Avoid adding extra details or over-elaborating.
- The question must explicitly contain {e1}.
- The answer of the generated question must be {e2}.
- If no direct question can be formed that includes {e1} and has {e2} as the answer based only on the provided documents, output exactly: CAN'T GENERATE
- Do not explain or output anything else.

Input:

{e1}:
{document1}

{e2}:
{document2}

Output:
Generated question containing {e1} with {e2} as the answer:
"""

SYSTEM_PROMPT_MISTRAL = """
You are given a database schema in CSV format describing several tables. Each schema entry includes:

original_column_name, column_name, column_description, data_format, value_description

You are also given a relationship triplet of the form:  
"Subject" "Predicate" "Object"

Your task is to determine whether this relationship can be explicitly or implicitly inferred from the schema.

Before answering, carefully reason through the schema:  
- Check if the predicate (relationship type) appears in any column names or descriptions.  
  - If the predicate is not an exact match but closely aligns with a related concept (e.g., "authored" vs. "worked on", "written by" vs. "created by"), you may infer the relationship using that column.  
  - However, do not infer if the predicate and available columns clearly indicate different roles (e.g., "executive producer" vs. "director"). Use common sense to distinguish such mismatches.
- Determine whether any tables or columns contain values that could represent the subject and object.  
- Decide whether an SQL query could be constructed using the schema to confirm the relationship.

Do not assume anything. The schema is the ground truth.  
- Use original column names when writing SQL.  
- Derive table names from the CSV filenames (e.g., director.csv -> director).

Return your answer as a JSON object with:
- "reasoning": an explanation of your decision based on the schema
- "canAnswer": true or false
- "SQL": the SQL query if canAnswer is true; if the table name contains hyphen (-) use backticks (`) to enclose the table name, otherwise an empty string

Schema:
{schema}

Input: "Clyde Geronimi" "director" "Cinderella"  
Output: {
  "reasoning": "The 'director.csv' file contains a 'director' column and a 'name' column, where 'name' refers to the movie name. The relationship asks whether 'Clyde Geronimi' directed 'Cinderella', which can be answered by checking if there is a row with director = 'Clyde Geronimi' and name = 'Cinderella'.",
  "canAnswer": true,
  "SQL": "SELECT * FROM director WHERE director = 'Clyde Geronimi' AND name = 'Cinderella';"
}

Input: "Ben Sharpsteen" "executive producer" "Alice in Wonderland"  
Output: {
  "reasoning": "Although the 'director.csv' file contains the name of the director and the movie, the schema does not mention the role of 'executive producer' in any column or description. Therefore, this relationship cannot be confirmed from the available schema.",
  "canAnswer": false,
  "SQL": ""
}

Input: 
"""


ENHANCE_MULTI_TURN = """You are given a multi-turn conversation where each new question builds upon information introduced in earlier turns.

Your task is to rewrite the final question so that it flows naturally in the context of the conversation, just like a human would continue a chat.

Instructions:

- Make the follow-up question sound natural by using pronouns or context-aware references instead of repeating full names, titles, or information already introduced.
- Use phrases like "the one we just talked about", he, she, etc., to make the conversation feel more fluid and coherent.
- Use the answer from the previous turn to help decide how to phrase the follow-up naturally.
- **Preserve the full meaning and specificity** of the original question.  Never add, remove, or infer any new information.
- If the original question is too long or sounds robotic, simplify it — as long as the meaning stays the same.


<example>
{example}
response:
<understand>
"Alice in Wonderland" with a natural reference like "this movie" since it was just mentioned in the answer to the previous question. The question can be simplified by replacing best friend of Mad Hatter to Mad Hatter's best friend.
</understand>
<new_query>
Who was the voice actor of Mad Hatter's best friend in this movie?
</new_query>
</example>

Now generate for this multi-turn conversation:
{input}

Now please generate following below format:
<understand>
Summarize what you're doing and what decision you made in plain language.
</understand>

<new_query>
Write the final rephrased question here. Make sure it flows naturally in the conversation. Never include {answer} in the the new query. Refer to {answer} using appropriate pronoun or reference. Never add or subtract any information in the original question. Only changes in the grammatical structure and the use of pronouns/references are allowed.
</new_query>
"""

MULTI_HOP_JUDGE = """
Given a merged question and a series of questions, verify that the merged question asks what the series of the questions are asking sequentially. 
The final answer for the merged question and the series of question should be the same: Use these instructions to check whether they are equivalent or not:

1. The merged question should retain all the important entities present in the series of question. The merged question is formed by the stitching of the questions with the answers as links so the answers should not be present in the merged questions.
2. The answer that is being prompted should be the same in both the cases. **There should not be a difference in what is being asked (nothing more or less) between the merged query and the last question of series of question.** If there is a difference rephrase the merged question (only grammatical changes) without removing any information so that the final answer asked is same as the series of question.
3. For each answer and the next question, check the type of the entity (the entity is same as the answer present in the question) connecting the two. The type (name, place, film, character) for each answer in the series of question should be the same as type of the same entity mentioned in the next question in the series of questions. Identify the types of each answer in the series of questions and check that the same entity being referred to in the next question is of the same type. If there is a change then that should be mentioned in the merged question. 
For example if Answer 1 is of type person and the type of Answer 1 in the next question is movie then the chaining should be phrased like "the movie having the same name as the person". **Do this for each answer, there shouldn't be any type mismatch**.

<example>
Merged query: Where was the person who voiced Lady Tremaine in the 1950 film in which there is a glass shoe born?
Series of question and answers:
Q1: Name the main character of Disney's movie in which there is a glass shoe. 
A1: Cinderella
Q2: Who voiced Lady Tremaine in the 1950 film Cinderella? 
A2: Eleanor Audley
Q3: Where was Eleanor Audley born?
Response:
<reasoning>
1. The final merged question retains all the important entities like Lady Tremaine, 1950, Disney, glass shoe. The answers Eleanor Audley and Cinderella are not present in the merged question which is accurate.
2. The final answer prompted in the last question is birth-place of Eleanor Audley and it is also the birth-place of Eleanor Audley in the merged query. Nothing more or less.
3. For each answer:
a. Cinderella: 
Type in A1: character
Type in Q2: film
Judge: Type Mismatch 
b. Eleanor Audley
Type in A2: person
Type in Q3: person
Judge: Correct type
</reasoning>
<final_verdict>
mismatch
</final_verdict>
<changes_needed>
the type has to be specified for Cinderella which is character and film while merging Q2 and Q3. The type of others answers match. The final answers also match.
</changes_needed>
<new_query>
Where was the voice actor of Lady Tremaine in the 1950 film named after the Disney's main character with the glass slipper born?
</new_query>
</example>

Give reasoning first and then provide final verdict as match/mismatch (if any one of 1, 2, or 3 fail then it is a mismatch). And if the final_verdict is mismatch, give the new question based on your reasoning.
Do this for the following now in the format of the above:
"""