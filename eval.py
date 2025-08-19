from unsloth import FastLanguageModel
from transformers import TextStreamer, GenerationConfig
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import numpy as np
import json
import argparse
from chat_template import *
import logging
import readline
import random

def get_args() -> argparse.Namespace:
    """
    Parses and returns the values of command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, required=True, help='Model name/trained LoRA adapters')
    parser.add_argument('--ctx-size', '--max-seq-len', type=int, dest='max_seq_len', default=2048, help='Context size of the model (default: 2048)')
    parser.add_argument('--max-new-tokens', type=int, default=2048, help='Max new tokens generated (default: 2048)')
    parser.add_argument('--dtype', type=str, default=None, help='Data type of the model\'s weights (default: None)')
    parser.add_argument('--load-in-4bit', type=bool, default=True, help='Load the model in 4-bit quantization (default: True)')
    parser.add_argument('--embedding-model', type=str, required=True, help='Embedding model')
    parser.add_argument('--base-url', type=str, default=None, help='API endpoint for accessing embedding model')
    parser.add_argument('--api-key', type=str, default='', help='API key for accessing embedding model')
    parser.add_argument('-d', '--data', type=str, required=True, help='Embeddings of the discussion threads')
    parser.add_argument('-p', '--prompt', type=str, required=False, help='Prompt passed into the model')
    parser.add_argument('--num-discussions', type=int, required=True, help='Number of discussion threads to retrieve for each prompt')
    parser.add_argument('-t', '--temp', type=float, default=0.5, help='Temperature (default: 0.5)')
    parser.add_argument('--top-k', type=int, default=40, help='Top k sampling (default: 40)')
    parser.add_argument('--min-p', type=float, default=0.05, help='Min p sampling (default: 0.01)')
    parser.add_argument('--top-p', type=float, default=0.95, help='Top p sampling (default: 0.95)')
    parser.add_argument('--repeat-penalty', type=float, default=1.1, help='Repetition penalty (default: 1.1)')

    args = parser.parse_args()
    return args

def load_data(path: str) -> list:

    """
    Loads the JSONL file and returns a list of dictionaries.
    """
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                json_obj = json.loads(line.strip())
                data.append(json_obj)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line: {line.strip()} - {e}")
    return data

class EmbeddingClient(OpenAI):
    """
    Extended OpenAI class for embedding queries.
    """
    def __init__(self, embedding_model: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.embedding_model = embedding_model

    def generate_embeddings(self, text: str) -> list[float]:
        """
        Generates embeddings for a text.
        """
        response = self.embeddings.create(model=self.embedding_model, input=text)
        return response.data[0].embedding
    
class DataStore:
    """
    Data store for containing discussion threads and embeddings.
    """
    def __init__(self, data: list[dict], embedding_client: EmbeddingClient):
        self.data = data
        self.client = embedding_client
    
    def top_k_threads(self, query: str, k: int) -> list[dict]:
        """
        Retrieves the top-k threads most relevant to the query.
        """
        query_embeddings = np.array(self.client.generate_embeddings(query))
        data_embeddings = np.array([x['embeddings'] for x in self.data])

        scores = cosine_similarity(query_embeddings.reshape(1, -1), data_embeddings)[0]
        sorted_data = sorted(zip(self.data, scores), key=lambda x: x[1], reverse=True)

        return [{'query': data['query'], 'thread': data['thread'], 'score': score} 
                for data, score in sorted_data[:k]]

def format_prompt_context(prompt: str, data: list[dict]) -> str:
    """
    Concatenates prompt and context into fine-tuned user prompt format.
    """
    # random.shuffle(data)
    return '\n\n'.join([f'<thread{i+1}>\n{x['thread']}\n</thread{i+1}>' for i, x in enumerate(data)] + [f'Question: {prompt}'])

def generate_response(prompt: str, data: list[dict]) -> None:
    """
    Formats the prompt and context and generates response.
    """
    # Format the prompt
    formatted_prompt = format_prompt_context(prompt, data)

    messages = [
        {"role": "user", "content": formatted_prompt},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        tokenize=True,
    ).to(model.device)

    # Generate response
    _ = model.generate(
        **inputs, 
        generation_config=generation_config,
        streamer=TextStreamer(tokenizer),
        do_sample=True,
        use_cache=True
    )

if __name__ == '__main__':
    args = get_args()

    # Load model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model, # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length=args.max_seq_len,
        dtype=args.dtype,
        load_in_4bit=args.load_in_4bit,
    )

    # Initialize client for embedding model
    client = EmbeddingClient(
        embedding_model=args.embedding_model,
        base_url=args.base_url, 
        api_key=args.api_key
    )

    # Load embedding data
    data_store = DataStore(load_data(args.data), client)

    # Generation config
    generation_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temp,
        top_k=args.top_k,
        min_p=args.min_p,
        top_p=args.top_p,
        repetition_penalty=args.repeat_penalty
    )

    # Generate specified prompt and exit
    if args.prompt:
        # Get top k most relevant discussion threads
        retrieved_threads = data_store.top_k_threads(args.prompt, args.num_discussions)

        generate_response(args.prompt, retrieved_threads)
    # Keep running until Ctrl+D is sent
    else:
        while True:
            try:
                prompt = input('>>> ').strip()
                if prompt == '':
                    print()
                    continue
                
                # Get top k most relevant discussion threads
                retrieved_threads = data_store.top_k_threads(prompt, args.num_discussions)

                # Generate the response
                generate_response(prompt, retrieved_threads)
            except KeyboardInterrupt:
                print()
                continue
            except EOFError:
                print()
                break
