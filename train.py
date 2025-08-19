import os
from unsloth import FastLanguageModel
import torch
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset
from transformers import HfArgumentParser
from dataclasses import dataclass, field
import logging
import os
import traceback
from chat_template import *

logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.INFO)

os.environ['UNSLOTH_STABLE_DOWNLOADS'] = '1'
os.environ['WANDB_PROJECT'] = os.environ.get('WANDB_PROJECT', 'reddit-reasoning')
os.environ['WANDB_LOG_MODEL'] = os.environ.get('WANDB_LOG_MODEL', 'checkpoint')

@dataclass
class Arguments:
    """
    Arguments for finetuning
    """
    model: str = field(metadata={'help': 'Model name'})
    max_seq_len: int = field(metadata={'help': 'Maximum sequence length/context size'})
    dataset: str = field(metadata={'help': 'Path of the dataset file'})
    save_dir: str = field(metadata={'help': 'Path to save the LoRA adapter'})
    dtype: str | None = field(default=None, metadata={'help': 'Data type of the model\'s weights.'})
    load_in_4bit: bool = field(default=True, metadata={'help': 'Loads the model in 4-bit quantization.'})
    fast_inference: bool = field(default=True, metadata={'help': 'Supports fast inferencing using vLLM.'})
    full_finetuning: bool = field(default=False, metadata={'help': 'Enables full-finetuning.'})
    gpu_mem_util: float = field(default=0.6, metadata={'help': 'Proportion of GPU memory to be utilized.'})
    # target_modules: str = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 
    #                        'gate_proj', 'up_proj', 'down_proj']
    lora_rank: int = field(default=32, metadata={'help': 'LoRA rank.'})
    lora_alpha: int | None = field(default=None, metadata={'help': 'LoRA alpha. If not specified, defaults to LoRA rank.'})
    lora_dropout: float = field(default=0, metadata={'help': 'LoRA dropout'})
    bias: str = field(default='none', metadata={'help': 'Model bias'})
    model_seed: int = field(default=3407, metadata={'help': 'Seed to load the LoRA adapter.'})
    use_rslora: bool = field(default=False, metadata={'help': 'Uses rank stabilized LoRA.'})
    shuffle_seed: int = field(default=42, metadata={'help': 'Seed to shuffle dataset.'})
    split: float = field(default=0.1, metadata={'help': 'Proportion to split train/test data.'})

    def __post_init__(self):
        if not self.lora_alpha:
            self.lora_alpha = self.lora_rank

def format_prompts(examples):
    """
    Format the prompt of each example in the dataset.
    """
    questions = examples['question']
    reasonings = examples['reasoning']
    answers = examples['answer']

    texts = []
    for question, reasoning, answer in zip(questions, reasonings, answers):
        text = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': question},
            {'role': 'assistant', 'content': REASONING_START + reasoning + REASONING_END + SOLUTION_START + answer + SOLUTION_END}
        ]
        texts.append(tokenizer.apply_chat_template(text, tokenize=False, add_generation_prompt=False))
    
    return {'text': texts}

if __name__ == '__main__':
    # args, remaining_args = get_args()
    hf_parser = HfArgumentParser((Arguments, SFTConfig))
    args, sft_config = hf_parser.parse_args_into_dataclasses()

    # references:
    # https://colab.research.google.com/drive/1Ys44kVvmeZtnICzWz0xgpRnrIOjZAuxp?usp=sharing
    
    # Load the model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        dtype=args.dtype,
        max_seq_length=args.max_seq_len,
        load_in_4bit=args.load_in_4bit,
        fast_inference=args.fast_inference,
        max_lora_rank=args.lora_rank,
        full_finetuning=args.full_finetuning,
        gpu_memory_utilization=args.gpu_mem_util
    )

    # Load LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 
                        'gate_proj', 'up_proj', 'down_proj'],
        lora_alpha=(args.lora_alpha or args.lora_rank),
        lora_dropout=args.lora_dropout,
        bias=args.bias,
        use_gradient_checkpointing='unsloth',
        random_state=args.model_seed,
        use_rslora=args.use_rslora,
        loftq_config=None
    )
    
    # Use custom chat template for base models
    if not tokenizer.chat_template:
        tokenizer.chat_template = chat_template

    logging.info(f'Loading dataset')
    # Load data
    dataset = load_dataset('json', data_files=args.dataset, split='train')
    # Shuffle and format the dataset
    dataset = dataset.shuffle(seed=args.shuffle_seed)
    dataset = dataset.map(format_prompts, batched=True)

    # Split the dataset into train/test data
    splits = dataset.train_test_split(test_size=0.1)
    train_ds = splits['train']
    eval_ds = splits['test']

    logging.info(f'Train dataset size: {len(train_ds)}')
    logging.info(f'Eval dataset size: {len(eval_ds)}')

    # Initialize SFT trainer
    # sft_config.seed = args.seed
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset=train_ds,
        eval_dataset=(eval_ds if len(eval_ds) > 0 else None),
        max_seq_length=args.max_seq_len,
        args=sft_config
    )

    # Show current memory stats
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    logging.info(f'GPU = {gpu_stats.name}. Max memory = {max_memory} GB.')
    logging.info(f'{start_gpu_memory} GB of memory reserved.')

    # Fine-tune the model
    try:
        trainer_stats = trainer.train(resume_from_checkpoint=(True if sft_config.resume_from_checkpoint else False))

        # Show final memory and time stats
        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
        used_percentage = round(used_memory / max_memory * 100, 3)
        lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
        logging.info(f'{trainer_stats.metrics['train_runtime']} seconds used for training.')
        logging.info(
            f'{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.'
        )
        logging.info(f'Peak reserved memory = {used_memory} GB.')
        logging.info(f'Peak reserved memory for training = {used_memory_for_lora} GB.')
        logging.info(f'Peak reserved memory % of max memory = {used_percentage} %.')
        logging.info(f'Peak reserved memory for training % of max memory = {lora_percentage} %.')
    except (KeyboardInterrupt, Exception) as e:
        traceback.print_exc()
        logging.error(e)
    finally:
        # Save adapter
        trainer.save_model(args.save_dir)
        trainer.save_state()

        logging.info(f'Saved model to \'{args.save_dir}\'')