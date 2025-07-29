from math import floor
import pandas as pd
import os
from transformers import AutoTokenizer
import json
import csv
import nltk
import heapq
import re
from deepmultilingualpunctuation import PunctuationModel
import time
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--directory', type=str, required=True, help='Directory containing the filtered CSV files')
parser.add_argument('--subreddit', type=str, required=True, help='Subreddit name')
parser.add_argument('--max-depth', type=str, default=3, help='Maximum depth of discussion thread')
parser.add_argument('--max-replies', type=str, default=3, help='Maxmum number of replies of a single discussion thread')
parser.add_argument('--ft-model', type=str, required=True, help='Model to be fine-tuned (its tokenizer is used for token counts)')
parser.add_argument('--gen-model', type=str, required=True, help='Model used to generate synthetic data (its tokenizer is used for token counts)')
parser.add_argument('--ft-max-tok', type=int, default=4000, help='Maximum number of tokens from a single discussion thread to include in fine-tuning')
parser.add_argument('--gen-max-tok', type=int, default=10000, help='Maximum number of tokens from a single discussion thread for generating synthetic data')

def check_depth(idx: int) -> None:
    """
    Get depth level of each comment. Top-level comment begins at 1
    """
    # Get parent id and index
    parent_id = comment_df.loc[idx, 'parent_id']
    parent_idx = comment_df.loc[idx, 'parent_idx']

    # Parent is a submission
    if parent_id.startswith('t3'):
        comment_df.loc[idx, 'depth'] = 1
    else:
        parent_depth = comment_df.loc[parent_idx, 'depth']
        comment_df.loc[idx, 'depth'] = parent_depth + 1

def build_submission_dict(id: str, author: str, title: str, body: str, flair: str | None = None) -> dict:
    return {
        'id': id,
        'author': author,
        'title': title,
    } | ({'flair': flair} if flair else {}) | {'body': body}

def build_comment_dict(id: str, parent_id: str, author: str, body: str) -> dict:
    return {
        'id': id,
        'parent_id': parent_id,
        'author': author,
        'body': body
    }

def json_token_count(data: dict, tokenizer) -> int:
    # Convert dictionary to string
    json_string = json.dumps(data, ensure_ascii=False)
    token_ids = tokenizer.encode(json_string)
    return len(token_ids)

def string_token_count(data: str, tokenizer) -> int:
    token_ids = tokenizer.encode(json.dumps(data, ensure_ascii=False))
    return len(token_ids)

cached_summaries = {}
def summarize_text(id: str, text: str, max_tokens: int, tokenizer) -> str:
    # Check if message is already summarized
    if id in cached_summaries:
        heap = cached_summaries[id].copy()
    else:
        # Text summarization implementation inherited from
        # https://www.kaggle.com/code/imkrkannan/text-summarization-with-nltk-in-python

        # Formatted text without special characters and digits
        formatted_text = re.sub('[^a-zA-Z]', ' ', text)
        formatted_text = re.sub(r'\s+', ' ', formatted_text)

        # Convert text to sentences
        sentence_list = nltk.sent_tokenize(text)
        refined_sentence_list = []

        # Fix punctuation of every sentence
        for sent in sentence_list:
            # Skip empty strings
            if not sent.strip() or not punc_model.preprocess(sent):
                continue
            fixed_sent = punc_model.restore_punctuation(sent)
            refined_sentence_list.extend(nltk.sent_tokenize(fixed_sent))

        sentence_list.clear()
        sentence_list = refined_sentence_list.copy()
        refined_sentence_list.clear()

        # Get word frequencies
        word_frequencies = {}
        # Loop through all words in formatted text
        for word in nltk.word_tokenize(formatted_text):
            # Check if word is a stopword
            if word in stopwords:
                continue
            # Increment frequency of the word
            if not word in word_frequencies:
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1
        
        max_frequency = max(word_frequencies.values())
        # Loop through the frequencies of each word and take weighted frequency
        for word in word_frequencies:
            word_frequencies[word] /= max_frequency

        sentence_scores = {}
        # Loop through each sentence
        for sent in sentence_list:
            # Tokenizer sentence into words
            words = nltk.word_tokenize(sent.lower())
            # Loop through each word
            for word in words:
                # Check if word exists in word frequencies
                if not word in word_frequencies:
                    continue
                # Assign sentence and its corresponding score
                if not sent in sentence_scores:
                    sentence_scores[sent] = word_frequencies[word]
                else:
                    sentence_scores[sent] += word_frequencies[word]
        
        # Build max-heap (-sentence_score, index, sentence)
        heap = [
            (-sentence_scores.get(sentence_list[i], float('inf')), i, sentence_list[i]) 
            for i in range(len(sentence_list))
        ]
        # Cache heap to avoid recomputing the scores
        cached_summaries[id] = heap.copy()
    
    heapq.heapify(heap)
    summary_sentences = []

    total_token = 0
    while len(heap) > 0:
        # Get index of sentences by highest score
        _, i, sent = heapq.heappop(heap)

        # Get token count of each sentence
        sent_tok_count = string_token_count(sent, tokenizer)
        # Check if token count exceeds maximum
        if total_token + sent_tok_count > max_tokens:
            break

        # Append index and sentence to list
        summary_sentences.append((i, sent))

        # Increment total token count
        total_token += sent_tok_count

    # Restore original order of the sentences
    summary_sentences.sort()

    summary = ' '.join([s for _, s in summary_sentences])
    return summary

def sort_top_comments(comment: tuple[int, str]) -> tuple[int, int]:
    idx, id = comment
    score = comment_df.loc[idx, 'score']
    num_replies = len(child_id_list[id])

    return (-score, -num_replies)

def truncate_msgs(msgs: list, total_json_tok: int, max_tok_limit: int, tokenizer) -> None:
    # Get token count for each message text and form a tuple (msg, msg token count)
    msg_tok_list = [
        {
            'msg': msg, 
            'tok_count': string_token_count(msg['body'], tokenizer),
            'old_tok_count': string_token_count(msg['body'], tokenizer)
        }
        for msg in msgs
    ]
    # Sort the list by longest message text
    msg_tok_list.sort(key=lambda x: x['tok_count'], reverse=True)

    # Get token count of the text in all messages in the thread
    total_text_tok = sum([item['tok_count'] for item in msg_tok_list])
    # Calculate token overhead count
    overhead_tok = total_json_tok - total_text_tok
    # Get max token count for the text of all messages
    max_text_tok = max(1, max_tok_limit - overhead_tok)
    # Get text token count that has overflowed the limit
    overflow_tok = total_text_tok - max_text_tok

    i = 0
    num_msgs = len(msg_tok_list)
    while i < num_msgs - 1:
        # Get token count difference between current message and next message
        num_tok_diff = msg_tok_list[i]['tok_count'] - msg_tok_list[i+1]['tok_count']
        # Get number of tokens to truncate
        # If `overflow_tok > num_tok_diff * (i+1)`, truncate top i+1 messages by `num_tok_diff`
        if overflow_tok > num_tok_diff * (i + 1):
            for j in range(i + 1):
                msg_tok_list[j]['tok_count'] -= num_tok_diff
            # Update overflow token count
            overflow_tok -= num_tok_diff * (i + 1)
            i += 1
        # If `overflow_tok <= num_tok_diff * (i+1)`, partially truncate top i+1 messages by `overflow_tok / (i+1)`
        else:
            partial_tok_trunc = overflow_tok / (i + 1)
            for j in range(i + 1):
                msg_tok_list[j]['tok_count'] -= partial_tok_trunc
            # Update overflow token count to 0
            overflow_tok = 0
            break
    
    # If there is still token overflow, uniformly truncate all messages by `overflow_tok // n`
    if overflow_tok > 0:
        uni_tok_trunc = overflow_tok / num_msgs
        for i in range(num_msgs):
            msg_tok_list[i]['tok_count'] -= uni_tok_trunc
    
    # Take floor of token cap for each message
    for i in range(num_msgs):
        msg_tok_list[i]['tok_count'] = max(1, int(msg_tok_list[i]['tok_count']))

    for item in msg_tok_list:
        msg, msg_tok_cap, old_tok_count = item['msg'], item['tok_count'], item['old_tok_count']
        # Message text does not require truncating
        if msg_tok_cap == old_tok_count:
            continue
        # Truncate each message by its corresponding token cap
        msg['body'] = summarize_text(msg['id'], msg['body'], msg_tok_cap, tokenizer)

def traverse_thread(idx: int, id: str, root: dict, thread: list) -> None:
    # Get id of each child
    child_ids = child_id_list[id]
    # Submission/comment has no children/replies
    if len(child_ids) == 0:
        return
    
    pbar.set_postfix({
        'post': root['post']['id'] if root.get('post') else id,
        'id': id,
        'depth': len(thread)
    })
    
    # Sort the children by most score, and then by most replies
    child_ids.sort(key=sort_top_comments)

    output_data = [id]

    is_submission = id.startswith('t3')
    # Current message is submission
    if is_submission:
        post = submission_df.iloc[idx]
        # Get flair text if exists
        flair = post['link_flair_text']
        flair = None if pd.isna(flair) else flair

        # Add submission data to root
        post_dict = build_submission_dict(id, post['author'], post['title'], post['body'], flair)
        root['post'] = post_dict

        # Add to thread
        thread.append(post_dict)

        post_dict['comments'] = []
        # Add top k children
        for child_idx, child_id in child_ids[:max_children]:
            child = comment_df.iloc[child_idx]
            post_dict['comments'].append(build_comment_dict(child_id, id, child['author'], child['body']))
        
        # Save text of all messages in the thread
        original_root_text = post_dict['body']
        original_children_text = [child['body'] for child in post_dict['comments']]

        # Get total token count of full thread using tokenizer for fine-tuning
        total_json_tok = json_token_count(root, ft_tokenizer)

        # Total number of tokens exceed max token count per thread to include in fine-tuning
        if total_json_tok > ft_max_tok:
            truncate_msgs(thread+post_dict['comments'], total_json_tok, ft_max_tok, ft_tokenizer)
        
        # Prepare full thread for output data
        output_data.append(json.dumps(root, ensure_ascii=False))

        # Restore original text to all messages in the thread
        post_dict['body'] = original_root_text
        for i, comment in enumerate(post_dict['comments']):
            comment['body'] = original_children_text[i]
        
        # Get total token count of full thread using tokenizer for synthetic data generation
        total_json_tok = json_token_count(root, gen_tokenizer)

        # Total number of tokens exceed max token count per thread for synthetic data generation
        if total_json_tok > gen_max_tok:
            truncate_msgs(thread+post_dict['comments'], total_json_tok, gen_max_tok, gen_tokenizer)
        
        # Prepare subthread for output data
        output_data.append(json.dumps(root, ensure_ascii=False))
        # No subthread content
        output_data.append(str(None).encode("utf-8", errors='replace').decode())

        # Restore original text to all messages in the thread
        post_dict['body'] = original_root_text
        for i, comment in enumerate(post_dict['comments']):
            comment['body'] = original_children_text[i]
        
        # Remove all children
        post_dict['comments'].clear()

        del original_root_text
        # original_children_text.clear()
    # Current message is comment
    else:
        parent = thread[-1]
        # Create comment data
        curr_comment = comment_df.iloc[idx]
        comment_dict = build_comment_dict(
            id, curr_comment['parent_id'], curr_comment['author'], curr_comment['body']
        )
        # Add to thread and parent's `comments` list
        parent['comments'].append(comment_dict)
        thread.append(comment_dict)

        # Add top k children
        comment_dict['comments'] = []
        for child_idx, child_id in child_ids[:max_children]:
            child = comment_df.iloc[child_idx]
            comment_dict['comments'].append(build_comment_dict(child_id, id, child['author'], child['body']))

        # Save original text of all messages in the thread
        original_thread_text = [msg['body'] for msg in thread]
        original_children_text = [child['body'] for child in comment_dict['comments']]

        # Get total token count of full thread
        total_json_tok = json_token_count(root, ft_tokenizer)

        # Total number of tokens exceeds max token count per thread to include in fine-tuning
        if total_json_tok > ft_max_tok:
            truncate_msgs(thread+comment_dict['comments'], total_json_tok, ft_max_tok, ft_tokenizer)

        # Prepare full thread for output data
        output_data.append(json.dumps(root, ensure_ascii=False))

        # Restore original text of all messages in the thread
        for i, msg in enumerate(thread):
            msg['body'] = original_thread_text[i]
        for i, child in enumerate(comment_dict['comments']):
            child['body'] = original_children_text[i]
        
        # Replace current comment with placeholder comment from parent's `comments` list
        # This is for the context of the subthread which excludes the text of the current comment
        placeholder_comment_dict = build_comment_dict(id, curr_comment['parent_id'], curr_comment['author'], '...')
        parent['comments'][0] = placeholder_comment_dict

        # Get total token count of full thread
        total_json_tok = json_token_count(root, gen_tokenizer) + json_token_count(comment_dict, gen_tokenizer)
        
        # Total number of tokens exceeds max token count per thread for synthetic data generation
        if total_json_tok > gen_max_tok:
            truncate_msgs(thread+comment_dict['comments'], total_json_tok, gen_max_tok, gen_tokenizer)
            
        # Prepare subthread and subthread context for output data
        output_data.append(json.dumps(comment_dict, ensure_ascii=False))
        output_data.append(json.dumps(root, ensure_ascii=False))

        # Restore original text to all messages in the thread
        for i, msg in enumerate(thread):
            msg['body'] = original_thread_text[i]
        for i, child in enumerate(comment_dict['comments']):
            child['body'] = original_children_text[i]

        # Restore parent's `comment` list to contain current comment
        parent['comments'][0] = comment_dict
        
        # Remove all children
        comment_dict['comments'].clear()

        # Cleanup original text of all messages in the thread
        original_thread_text.clear()
        del original_thread_text
        # original_children_text.clear()
    
    # Cleanup original text of children
    original_children_text.clear()
    del original_children_text

    # Write data to CSV
    writer.writerow(output_data)
    output_data.clear()
    del output_data

    # Traverse down the threads
    for child_idx, child_id in child_ids:
        traverse_thread(child_idx, child_id, root, thread)

    # Remove current message from thread
    thread.pop()
    # Clear current comment from parent `comments` list
    if not is_submission:
        parent['comments'].clear()

if __name__ == '__main__':
    args = parser.parse_args()
    subreddit = args.subreddit
    directory = args.directory
    path = os.path.join(directory, subreddit)

    max_depth = args.max_depth
    max_children = args.max_replies
    ft_max_tok = args.ft_max_tok # Max token per thread to include in fine-tuning
    gen_max_tok = args.gen_max_tok # Max token per thread for synthetic data generation
    if ft_max_tok < 1000 or gen_max_tok < 1000:
        raise ValueError(f'ft_max_tok and/or gen_max_tok must be at least 1000 tokens')

    gen_model_id = args.gen_model
    gen_tokenizer = AutoTokenizer.from_pretrained(gen_model_id)

    ft_model_id = args.ft_model
    ft_tokenizer = AutoTokenizer.from_pretrained(ft_model_id)

    # Load NLTK libraries and punctuation model
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    stopwords = nltk.corpus.stopwords.words('english')
    punc_model = PunctuationModel()

    start_time = time.perf_counter()

    output_file = open(f'{path}_threads.csv', 'w', encoding='utf-8', newline="")
    writer = csv.writer(output_file)
    writer.writerow(['id', 'full_thread', 'subthread', 'subthread_context'])

    # Load submissions and comments into dataframes
    submission_df = pd.read_csv(f'{path}_filtered_submissions.csv', dtype={
        'author': 'string',
        'title': 'string',
        'body': 'string',
        'id': 'string',
        'link_flair_text': 'string',
    })
    submission_df['created'] = pd.to_datetime(submission_df['created'], format='%Y-%m-%d %H:%M:%S')

    comment_df = pd.read_csv(f'{path}_filtered_comments.csv', dtype={
        'author': 'string',
        'body': 'string',
        'id': 'string',
        'parent_id': 'string',
    })
    comment_df['created'] = pd.to_datetime(comment_df['created'], format='%Y-%m-%d %H:%M:%S')

    # Map each parent id to its index (will be efficient for parent lookup when getting depth level)
    id_to_idx = pd.Series(comment_df.index.values, index=comment_df['id'])
    comment_df['parent_idx'] = comment_df['parent_id'].map(id_to_idx).astype('Int64')
    
    comment_df['depth'] = pd.NA
    comment_df['depth'] = comment_df['depth'].astype('Int64')

    # Iterate each comment and get its depth level
    for idx, row in comment_df.iterrows():
        check_depth(idx)
    
    # Only keep messages with a depth level of at most `max_depth + 1` to include leaf comments 
    # as replies to comments with depth level of `max_depth`
    comment_df = comment_df[comment_df['depth'] <= max_depth + 1]
    comment_df = comment_df.reset_index(drop=True)

    # Get child ids of every submission/comment
    child_id_list = {}

    # Iterate through each submission
    for idx, row in submission_df.iterrows():
        child_id_list[row['id']] = []

    # Iterate through each comment
    for idx, row in comment_df.iterrows():
        # Get id and parent id
        id = row['id']
        parent_id = row['parent_id']
        # Add index and id into parent
        child_id_list[parent_id].append((idx, id))
        child_id_list[id] = []

    pbar = tqdm(submission_df.iterrows(), total=len(submission_df), dynamic_ncols=100, desc='Processing')
    # Traverse the threads starting from each submission
    for idx, row in pbar:
        root = {}
        thread = []

        traverse_thread(idx, row['id'], root, thread)
        # Clear cached summaries since they won't be accessed in the next submission
        cached_summaries.clear()

        del root
        del thread

    output_file.close()

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.6f} seconds")
