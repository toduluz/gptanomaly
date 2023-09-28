import multiprocessing
import time

from arguments import PretokenizationArguments
from datasets import load_dataset, concatenate_datasets

from transformers import AutoTokenizer, HfArgumentParser
import random
import copy

def tokenize(example):
    output = {}
    # print(tokenizer.tokenize(example["text"][:100]))
    output["input_ids"] = tokenizer(example["text"], truncation=False)["input_ids"]
    # output["ratio_char_token"] = len(example["content"]) / len(output["input_ids"])
    return output

def process_data(example):
    text = example['text']

    # Split the text by "|" delimiter
    text = text.split('|')
    
    # Combine the modified text back together
    text = ' '.join(text)
    example['text'] = text

    return example

def augment_data(example):
    text = example['text']

    # Split the text by "|" delimiter
    text = text.split('|')

    # Decide with 50% probability whether to split or reverse the text
    if random.random() < 0.5:
        random.shuffle(text)
    else:
        # Reverse the text
        text = text[::-1]

    # Combine the modified text back together
    text = ' '.join(text)

    example['text'] = text
    example['labels'] = 1

    return example


parser = HfArgumentParser(PretokenizationArguments)
args = parser.parse_args()
if args.num_workers is None:
    args.num_workers = multiprocessing.cpu_count()
tokenizer = AutoTokenizer.from_pretrained("gpt2-large", model_max_length=args.model_max_length)

t_start = time.time()
ds_train = load_dataset(
            'csv',
            data_files=args.data_dir + "/" + args.dataset_name + "/" + "/train.csv",
            split="train")
ds_validation = load_dataset(
            'csv',
            data_files=args.data_dir + "/" + args.dataset_name + "/" + "/validation.csv",
            split="train")
ds_test = load_dataset(
            'csv',
            data_files=args.data_dir + "/" + args.dataset_name + "/" + "/test.csv",
            split="train")
print(f"Dataset loaded in {time.time()-t_start:.2f}s")

t_start = time.time()
ds_train_augmented = copy.deepcopy(ds_train)
ds_train_orig = ds_train.map(
    process_data,
    num_proc=args.num_workers,
)
ds_train_augmented = ds_train_augmented.map(
    augment_data,
    num_proc=args.num_workers,
)

ds_train_augmented = ds_train_augmented.shuffle(seed=42)
ds_train_augmented = ds_train_augmented.filter(lambda example, idx: idx % 2 == 0, with_indices=True)
# Concatenate the original dataset and the augmented dataset
ds_train = concatenate_datasets([ds_train_orig, ds_train_augmented])
ds_train = ds_train.shuffle(seed=42)

ds_validation = ds_validation.map(
    process_data,
    num_proc=args.num_workers,
)
ds_test = ds_test.map(
    process_data,
    num_proc=args.num_workers,
)
print(f"Dataset tokenized in {time.time()-t_start:.2f}s")

t_start = time.time()
ds_train.save_to_disk("outputs/" + args.dataset_name + "/" + "train_dataset")
ds_validation.save_to_disk("outputs/" + args.dataset_name + "/" + "validation_dataset")
ds_test.save_to_disk("outputs/" + args.dataset_name + "/" + "test_dataset")
# ds.push_to_hub(args.tokenized_data_repo)
print(f"Data saved in {time.time()-t_start:.2f}s")
