import pandas as pd
import os
from openai import OpenAI
import random
import json
import re
import time
import argparse
from tqdm import tqdm
import logging
import traceback

logging.basicConfig(format='[%(levelname)s] %(message)s')
logger = logging.getLogger()

def get_args() -> argparse.Namespace:
    """
    Parses and returns the values of command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, required=True, help='CSV file path containing the discussion threads')
    parser.add_argument('-o', '--out', type=str, default='./', help='Directory path for saving the dataset and embeddings')
    parser.add_argument('--out-prefix', type=str, default=None, help='Prefix prepended to the file names of the dataset and embeddings')
    parser.add_argument('--base-url', type=str, default=None, help='API endpoint for accessing completion and embedding models')
    parser.add_argument('--api-key', type=str, default='', help='API key for accessing completion and embedding models')
    parser.add_argument('--completion-model', type=str, required=True, help='Completion model for generating question-answer pairs')
    parser.add_argument('--embedding-model', type=str, required=True, help='Embedding model to encode queries')
    parser.add_argument('--sys', type=str, default='You are a helpful assistant.', help='System prompt')
    parser.add_argument('--question-instructions', type=str, nargs='*', default=[], help='Additional instructions for generating the question')
    parser.add_argument('--answer-instructions', type=str, nargs='*', default=[], help='Additional instructions for generating the reasoning and answer')
    parser.add_argument('-p', '--p', type=float, default=1.0, help='Percentage of the dataset that includes the oracle discussion thread (default: 1.0)')
    parser.add_argument('--num-distractors', type=int, default=2, help='Number of distractor discussion thread to include for each data point (default: 2)')
    parser.add_argument('--num-questions', type=int, default=1, help='Number of question-answer pairs to generate for each discussion thread (default: 1)')
    parser.add_argument('--num-discussions', type=int, default=None, help='Number of discussion threads to generate question-answer pairs (default: None)')

    args = parser.parse_args()
    return args

class ExtendedClient(OpenAI):
    """
    Extended OpenAI class for embedding queries and generating question-answer pairs.
    """
    def __init__(self, completion_model: str, embedding_model: str, system_prompt: str, question_instructions: list[str], answer_instructions: list[str], *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.completion_model = completion_model
        self.embedding_model = embedding_model
        self.system_prompt = system_prompt
        self.question_instructions = question_instructions
        self.answer_instructions = answer_instructions

        # Regex pattern for parsing response
        self.question_pattern = re.compile(
            r"(?s)<question>(?P<question>.*?)(?:</question>|```|\Z)"
        )
        self.answer_pattern = re.compile(
            r"(?s)<reasoning>(?P<reasoning>.*?)</reasoning>\s*"
            r"(?P<answer>.*?)(?:```|\Z)"
        )
        
        self.question_prompt = f"""Given an excerpt from a Reddit discussion, write a question that corresponds to the content of this discussion.

Answer in this template/outline:
```xml
<question>
your question
</question>
```
If there is insufficient information in this discussion to form a question, respond with empty blocks in this template:
```xml
<question>
</question>
```
Additional notes:
- Question should NOT mention the discussion or explicitly include Reddit terms such as "comment", "post", "thread", "author", or "OP".\
{'\n' + '\n'.join(f'- {instruction}' for instruction in self.question_instructions) if self.question_instructions else ''}"""

        self.answer_prompt = f"""Given an excerpt from a Reddit discussion and a question, write an answer that corresponds to the content of this discussion, referencing only the post and comments.

In your reasoning, carefully review the post and every comment and walk through your thought process before answering (what parts of the discussion are meaningful and why).

Answer in this template:
```
<reasoning>
Let's break this down.

The question is asking...

The post discusses... It mentions...

(For each comment)
Comment 1 mentions...

Comment 2 mentions...

...

The key ideas presented in this discussion include...

Therefore, the answer should be about...
</reasoning>
your answer
```

Additional notes:
- Answer should NOT mention the discussion or explicitly include Reddit terms such as "comment", "post", "thread", "author", or "OP".\
{'\n' + '\n'.join(f'- {instruction}' for instruction in self.answer_instructions) if self.answer_instructions else ''}"""

    def generate_embeddings(self, text: str) -> list[float]:
        """
        Generates embeddings for a text.
        """
        response = self.embeddings.create(model=self.embedding_model, input=text)
        return response.data[0].embedding

    def generate_qa_pair(self, thread: str, thread_context: str | None = None) -> dict:
        """
        Takes in a JSON text of the discussion thread and, optionally, a JSON text of the discussion thread context
        and generates a question-answer pair based on the discussion thread.
        """
        if thread_context:
            user_prompt = f"""Here is the context containing the original post and any previous comments before the given Reddit comment. This information should only be used for understanding the context of the comment.
```xml
{thread_context}
```
Here is the Reddit comment:
```xml
{thread}
```"""
        else:
            user_prompt = f"""Here is the Reddit discussion:
```xml
{thread}
```"""
        # Generate the question
        match = None
        attempt = 0
        # Attempt three times before throwing an error
        while not match and attempt < 3:
            # Generate response and get content
            response = self.chat.completions.create(
                model=args.completion_model,
                messages=[
                    {'role': 'system', 'content': self.system_prompt},
                    {'role': 'user', 'content': self.question_prompt + '\n\n' + user_prompt}
                ]
            )
            content = response.choices[0].message.content

            # Parse question
            match = self.question_pattern.search(content)

            attempt += 1

        if not match:
            raise ValueError(f'expect <question> block in order:\n"{content}"')
        
        content_dict = {key: value.strip() for key, value in match.groupdict().items()}

        # Return blank if question is blank
        if content_dict['question'] == '':
            return content_dict | {'reasoning': '', 'answer': ''}

        # Now, generate the reasoning and answer
        match = None
        attempt = 0
        # Attempt three times before throwing an error
        while not match and attempt < 3:
            # Generate response and get content
            response = self.chat.completions.create(
                model=args.completion_model,
                messages=[
                    {'role': 'system', 'content': self.system_prompt},
                    {
                        'role': 'user', 
                        'content': f"""{self.answer_prompt}

{user_prompt}

Here is the question:
```xml
<question>
{content_dict['question']}
</question>
```"""
                    }
                ]
            )
            content = response.choices[0].message.content

            # Parse reasoning and answer
            match = self.answer_pattern.search(content)

            attempt += 1

        if not match:
            raise ValueError(f'expect <reasoning> and <answer> block in order:\n"{content}"')
        
        content_dict = content_dict | {key: value.strip() for key, value in match.groupdict().items()}
        return content_dict

def format_prompt(question: str, thread_list: list[str]) -> str:
    """
    Given a question and a list of discussion threads, formats a prompt for the fine-tuning dataset.
    """
    prompt = ''
    # Shuffle list
    random.shuffle(thread_list)
    for i, thread in enumerate(thread_list):
        prompt += f'<thread{i+1}>\n{thread}\n</thread{i+1}>\n\n'
    
    prompt += f'Question: {question}'
    return prompt

def format_answer(reasoning: str, answer: str) -> str:
    """
    Formats the reasoning and answer for the fine-tuning dataset.
    """
    return f'<think>{reasoning}</think>\n<answer>{answer}</answer>'

if __name__ == '__main__':
    # Get values of command line arguments
    args = get_args()
    if args.p > 1 or args.p < 0:
        raise ValueError('p must be in between 0 and 1')
    
    out_directory = args.out
    dataset_file_path = os.path.join(args.out, (f'{args.out_prefix}_' if args.out_prefix else '') + 'dataset.jsonl')
    embedding_file_path = os.path.join(args.out, (f'{args.out_prefix}_' if args.out_prefix else '') + 'embeddings.jsonl')

    os.makedirs(out_directory, exist_ok=True)
    dataset_file = open(dataset_file_path, 'w', encoding='utf-8')
    embedding_file = open(embedding_file_path, 'w', encoding='utf-8')

    # Initialize client for completion and embedding model
    client = ExtendedClient(
        completion_model=args.completion_model,
        embedding_model=args.embedding_model,
        system_prompt=args.sys,
        question_instructions=args.question_instructions,
        answer_instructions=args.answer_instructions,
        base_url=args.base_url, 
        api_key=args.api_key
    )

    discussion_df = pd.read_csv(args.file, dtype={
        'id': 'string',
        'post_id': 'string',
        'full_thread': 'string',
        'subthread': 'string',
        'subthread_context': 'string'
    })

    if args.num_distractors > (args.num_discussions or len(discussion_df)) - 1:
        raise ValueError('--num-distractors cannot exceed --num-discussions - 1')

    # Shuffle the dataframe
    discussion_df = discussion_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Keep first `num_discussions` discussion threads
    num_discussions = args.num_discussions or len(discussion_df)
    discussion_df = discussion_df.iloc[:num_discussions]
    df_len = len(discussion_df)

    # Mark p% of the data to include oracle discussion thread and (1-p)% of the data to exclude oracle discussion thread
    midpoint = int(len(discussion_df) * args.p)
    discussion_df['include_oracle'] = [True] * midpoint + [False] * (df_len - midpoint)

    # Shuffle the dataframe again if p is less than 1
    if args.p < 1:
        discussion_df = discussion_df.sample(frac=1, random_state=42).reset_index(drop=True)

    num_failed = 0
    num_skipped = 0
    pbar = tqdm(discussion_df.iterrows(), total=df_len, dynamic_ncols=100, desc='Generating')
    # Iterate through each discussion thread
    for idx, row in pbar:
        full_thread = row['full_thread']
        subthread = row['subthread']
        subthread_context = row['subthread_context'] if pd.notna(row['subthread_context']) else None
        include_oracle = row['include_oracle']
        
        # Generate `num_questions` questions for each discussion thread
        for j in range(args.num_questions):
            pbar.set_postfix({'question': f'{j}/{args.num_questions}', 'skipped': num_skipped, 'failed': num_failed})
            # Generate question-answer pair
            try:
                data = client.generate_qa_pair(subthread, subthread_context)
            except Exception as e:
                num_failed += 1
                traceback.print_exc()
                logger.error(e)
                pbar.set_postfix({'question': f'{j+1}/{args.num_questions}', 'skipped': num_skipped, 'failed': num_failed})
                continue
            

            question, reasoning, answer = data['question'], data['reasoning'], data['answer']
            # Skip discussion if any fields are empty
            if not (question and reasoning and answer):
                num_skipped += 1
                pbar.set_postfix({'question': f'{j+1}/{args.num_questions}', 'skipped': num_skipped, 'failed': num_failed})
                break

            # Randomly sample `num_distractor` indices for selecting distractor discussion threads
            distractor_indices = random.sample(list(set(range(df_len)) - {idx}), args.num_distractors)

            # Get distractor and oracle discussion threads
            distractor_threads = [discussion_df.loc[i, 'full_thread'] for i in distractor_indices]
            oracle_thread = [full_thread] if include_oracle else []

            # Format question
            formatted_question = format_prompt(question, distractor_threads + oracle_thread)
            # formatted_answer = format_answer(reasoning, answer)

            # Generate embeddings for the question
            try:
                embeddings = client.generate_embeddings(question)
            except Exception as e:
                num_failed += 1
                traceback.print_exc()
                logger.error(e)
                pbar.set_postfix({'question': f'{j+1}/{args.num_questions}', 'skipped': num_skipped, 'failed': num_failed})
                continue

            embedding_data = {
                'id': row['id'],
                'query': question,
                'thread': full_thread,
                'embeddings': embeddings
            }
            
            # Write question-answer pair and embeddings to files
            dataset_file.write(json.dumps({'question': formatted_question, 'reasoning': reasoning, 'answer': answer}) + '\n')
            embedding_file.write(json.dumps(embedding_data) + '\n')

            pbar.set_postfix({'question': f'{j+1}/{args.num_questions}', 'skipped': num_skipped, 'failed': num_failed})
    
    dataset_file.close()
    embedding_file.close()