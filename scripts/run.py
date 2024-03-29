#!/usr/bin/env python
# coding=utf-8
import argparse
import json
import logging
import math
import os
import random
from itertools import chain
from pathlib import Path
import shutil
import pickle
import pandas as pd

import datasets
import torch
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torch.nn as nn
import torch.nn.functional as F
from model import model
import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModel,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    SchedulerType,
    get_scheduler,
)
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import numpy as np


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.38.0.dev0")

logger = get_logger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a Masked Language Modeling task")
    parser.add_argument(
        "--train_path",
        type=str,
        required=True,
        help="Path to the training dataset",
    )
    parser.add_argument(
        "--test_path",
        type=str,
        required=True,
        help="Path to the test dataset",
    )
    parser.add_argument(
        "--validation_split_percentage",
        default=10,
        help="The percentage of the train set used as validation set in case there's no validation split",
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=None,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated."
        ),
    )
    parser.add_argument(
        "--line_by_line",
        type=bool,
        default=False,
        help="Whether distinct lines of text in the dataset are to be handled as distinct sequences.",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--mlm_probability", type=float, default=0.15, help="Ratio of tokens to mask for masked language modeling loss"
    )
    parser.add_argument(
        "--trust_remote_code",
        type=bool,
        default=False,
        help=(
            "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
            "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
            "execute code present on the Hub on your local machine."
        ),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations. '
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--low_cpu_mem_usage",
        action="store_true",
        help=(
            "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded. "
            "If passed, LLM loading time and RAM consumption will be benefited."
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="For debugging purposes or quicker training, truncate the number of training examples to this value if set.",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="For debugging purposes or quicker training, truncate the number of evaluation examples to this value if set.",
    )
    parser.add_argument(
        "--max_test_samples",
        type=int,
        default=None,
        help="For debugging purposes or quicker training, truncate the number of test examples to this value if set.",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=None,
        help="If passed, will stop training early when the metric monitored stops improving.",
    )
    parser.add_argument(
        "--max_save_limit",
        type=int,
        default=3,
        help="The maximum number of checkpoints to save.",
    )
    parser.add_argument(
        "--encoder_name_or_path",
        type=str,
        default=None,
        help="The name or path of the encoder model to use.",
    )
    parser.add_argument(
        "--normal_only",
        type=bool,
        default=True,
        help="If passed, will only use normal samples for training and testing.",
    )
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.train_path is not None:
        with open(args.train_path, "rb") as f:
            train_data = pickle.load(f)
            train_df = pd.DataFrame(train_data)
   
            # Check if 'label' column contains lists
            if train_df['Label'].apply(isinstance, args=(list,)).all():
                # Find the max of each list in the 'label' column
                train_df['Label'] = train_df['Label'].apply(max)
            train_df['EventTemplate'] = train_df['EventTemplate'].apply(lambda x: ' '.join([i.replace('<*>', '') for i in x]))
            if args.normal_only:
                train_df = train_df[train_df['Label'] == 0].reset_index(drop=True)
    if args.test_path is not None:
        with open(args.test_path, "rb") as f:
            test_data = pickle.load(f)
            test_df = pd.DataFrame(test_data)
            # Check if 'label' column contains lists
            if test_df['Label'].apply(isinstance, args=(list,)).all():
                # Find the max of each list in the 'label' column
                test_df['Label'] = test_df['Label'].apply(max)
            test_df['EventTemplate'] = test_df['EventTemplate'].apply(lambda x: ' '.join([i.replace('<*>', '') for i in x]))
    
    # Load the dataset
    raw_datasets = datasets.Dataset.from_pandas(train_df).train_test_split(
        test_size=args.validation_split_percentage / 100
    )
    raw_datasets["validation"] = raw_datasets["test"]
    raw_datasets["test"] = datasets.Dataset.from_pandas(test_df)
        
    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name, trust_remote_code=args.trust_remote_code)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
    
    if args.encoder_name_or_path:
        encoder_config = AutoConfig.from_pretrained(args.encoder_name_or_path, trust_remote_code=args.trust_remote_code)

    print("*" * 100)
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name, use_fast=not args.use_slow_tokenizer, trust_remote_code=args.trust_remote_code
        )
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, use_fast=not args.use_slow_tokenizer, trust_remote_code=args.trust_remote_code
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script. "
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    print("*" * 100)

    # model = model.LogRobertaAEModelForMaskedLM(config)
    if args.model_name_or_path:
        model = AutoModelForMaskedLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForMaskedLM.from_config(config, trust_remote_code=args.trust_remote_code)
    
    if args.encoder_name_or_path:
        encoder = AutoModel.from_pretrained(
            args.encoder_name_or_path,
            from_tf=bool(".ckpt" in args.encoder_name_or_path),
            config=encoder_config,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
            trust_remote_code=args.trust_remote_code,
        )

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = encoder.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        encoder.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["train"].column_names
    text_column_name = "EventTemplate" if "EventTemplate" in column_names else column_names[0]

    if args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            logger.warning(
                "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                " override this default with `--block_size xxx`."
            )
            max_seq_length = 1024
    else:
        if args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({args.max_seq_length}) is larger than the maximum length for the "
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)

    if args.line_by_line:
        # When using line_by_line, we just tokenize each nonempty line.
        padding = "max_length" if args.pad_to_max_length else False

        def preprocess_function(examples):
            batch = tokenizer(
                examples[text_column_name],
                padding=padding,
                truncation=True,
                max_length=max_seq_length,
                # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
                # receives the `special_tokens_mask`.
                return_special_tokens_mask=False,
            )
            batch['log_labels'] = examples['Label']
            return batch

        # def tokenize_function(examples):
        #     # Remove empty lines
        #     examples[text_column_name] = [
        #         line for line in examples[text_column_name] if len(line) > 0 and not line.isspace()
        #     ]
        #     return tokenizer(
        #         examples[text_column_name],
        #         padding=padding,
        #         truncation=True,
        #         max_length=max_seq_length,
        #         # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
        #         # receives the `special_tokens_mask`.
        #         return_special_tokens_mask=True,
        #     )

        with accelerator.main_process_first():
            tokenized_datasets = raw_datasets.map(
                preprocess_function,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=[*column_names],
                load_from_cache_file=not args.overwrite_cache,
                desc="Running tokenizer on dataset line_by_line",
            )
    else:
        # Otherwise, we tokenize every text, then concatenate them together before splitting them in smaller parts.
        # We use `return_special_tokens_mask=True` because DataCollatorForLanguageModeling (see below) is more
        # efficient when it receives the `special_tokens_mask`.
        def tokenize_function(examples):
            return tokenizer(examples[text_column_name], return_special_tokens_mask=True)

        with accelerator.main_process_first():
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not args.overwrite_cache,
                desc="Running tokenizer on every text in dataset",
            )

        # Main data processing function that will concatenate all texts from our dataset and generate chunks of
        # max_seq_length.
        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, and if the total_length < max_seq_length  we exclude this batch and return an empty dict.
            # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
            total_length = (total_length // max_seq_length) * max_seq_length
            # Split by chunks of max_len.
            result = {
                k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
                for k, t in concatenated_examples.items()
            }
            return result

        # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
        # remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
        # might be slower to preprocess.
        #
        # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
        # https://huggingface.co/docs/datasets/process#map

        with accelerator.main_process_first():
            tokenized_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                load_from_cache_file=not args.overwrite_cache,
                desc=f"Grouping texts in chunks of {max_seq_length}",
            )

    def precompute_logits(mlm_preds: torch.Tensor, k: int = 5) -> torch.Tensor:
        # Return the top-k
        return torch.topk(mlm_preds, k, dim=-1).indices
    
    def compute_metrics_for_test(mlm_preds: np.ndarray, mlm_labels: np.ndarray, secondary_logits: np.ndarray, log_labels: np.ndarray) -> dict:

        top_k_accuracies = []

        for preds, lbls in zip(mlm_preds, mlm_labels):
            # Remove -100 values
            mask = lbls != -100
            preds = preds[mask]
            lbls = lbls[mask]

            # Check if the true labels are in the top-k predictions
            correct = np.expand_dims(lbls, axis=-1) == preds

            # Compute the top-k accuracy for the current sentence
            top_k_accuracy = correct.any(axis=-1).mean()
            
            # Replace NaN values with 0
            top_k_accuracy = np.nan_to_num(top_k_accuracy, nan=0)

            top_k_accuracies.append(top_k_accuracy)

        # Calculate the average of each sample
        mean_of_each_sample = np.mean(secondary_logits, axis=1)
        print(mean_of_each_sample.shape)

        # Calculate the overall mean and standard deviation
        overall_mean = np.mean(mean_of_each_sample)
        overall_std_dev = np.std(mean_of_each_sample)
        print(overall_mean, overall_std_dev)

        # Calculate the distance of each sample's mean from the overall mean
        distances = np.abs(mean_of_each_sample - overall_mean) / overall_std_dev
        print(distances.shape)

        # Calculate the standard deviation of these distances
        std_dev_of_distances = np.std(distances)
        
        return find_best_threshold(top_k_accuracies, np.linspace(0.51, 1.0), distances, std_dev_of_distances, log_labels)
    
    def find_best_threshold(top_k_accuracies: list, accuracy_thresholds: np.ndarray, deviations: np.ndarray, std_dev, log_labels: np.ndarray) -> dict:
        best_threshold = 0
        best_f1 = 0
        best_precision = 0
        best_recall = 0
        best_roc_auc = 0
        best_d = 0

        for threshold in accuracy_thresholds:
            # Compute the predictions
            for d in range(1, 10):
                preds = (np.array(top_k_accuracies) < threshold) | (deviations > d * std_dev)

                # Compute the metrics
                f1 = f1_score(log_labels, preds)
                precision = precision_score(log_labels, preds)
                recall = recall_score(log_labels, preds)
                roc_auc = roc_auc_score(log_labels, preds)

                # Update the best metrics
                if f1 > best_f1:
                    best_d = d
                    best_threshold = threshold
                    best_f1 = f1
                    best_precision = precision
                    best_recall = recall
                    best_roc_auc = roc_auc
            
        return {
            "threshold": best_threshold,
            "d": best_d,
            "f1": best_f1,
            "precision": best_precision,
            "recall": best_recall,
            "roc_auc": best_roc_auc,
        }

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]
    test_dataset = tokenized_datasets["test"]
    if args.max_train_samples is not None:
        train_dataset = train_dataset.select(range(args.max_train_samples))
    if args.max_eval_samples is not None:
        eval_dataset = eval_dataset.select(range(args.max_eval_samples))
    if args.max_test_samples is not None:
        test_dataset = test_dataset.select(range(args.max_test_samples))

    # Conditional for small test subsets
    if len(train_dataset) > 3:
        # Log a few random samples from the training set:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Data collator
    # This one will take care of randomly masking the tokens.
    # data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=args.mlm_probability)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=args.mlm_probability)

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
    test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    # discriminator_parameters = optimizer_grouped_parameters + [
    #     {
    #         "params": discriminator_model.parameters(),
    #         "weight_decay": args.weight_decay,
    #     }
    # ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
    # shorter in multiprocess)
    print(len(train_dataloader))
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps if overrode_max_train_steps else args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    model, encoder, optimizer, train_dataloader, eval_dataloader, test_dataloader, lr_scheduler = accelerator.prepare(
        model, encoder, optimizer, train_dataloader, eval_dataloader, test_dataloader, lr_scheduler
    )

    # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
    # if accelerator.distributed_type == DistributedType.TPU:
    #     model.tie_weights()
    print(len(train_dataloader))
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("mlm_no_trainer", experiment_config)

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    best_loss = float("inf")
    best_epoch = None
        
    for epoch in range(starting_epoch, args.num_train_epochs):
        encoder.train()
        model.train()
        total_loss = 0
        if args.with_tracking:
            total_loss = 0
        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader
        
        for step, batch in enumerate(active_dataloader):
            with accelerator.accumulate(model):
                with torch.no_grad():
                    encodings = encoder(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])

                outputs = model(encodings.last_hidden_state, labels=batch['labels'])
                loss = outputs.loss
                # We keep track of the loss at each epoch
                total_loss += loss.detach().float()
                if args.with_tracking:
                    total_loss += loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

            if completed_steps >= args.max_train_steps:
                break
    
        train_loss = total_loss.item() / len(train_dataloader)
        logger.info(f"epoch: {epoch} train_loss: {train_loss}")

        encoder.eval()
        model.eval()
        losses = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                encodings = encoder(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
                outputs = model(encodings.last_hidden_state, labels=batch['labels'])
            loss = outputs.loss
            losses.append(accelerator.gather_for_metrics(loss.repeat(args.per_device_eval_batch_size)))

        losses = torch.cat(losses)
        eval_loss = torch.mean(losses)
        logger.info(f"epoch: {epoch} eval_loss: {eval_loss}")

        # if args.with_tracking:
        #     accelerator.log(
        #         {
        #             # "perplexity": perplexity,
        #             "eval_loss": eval_loss,
        #             "train_loss": total_loss.item() / len(train_dataloader),
        #             "epoch": epoch,
        #             "step": completed_steps,
        #         },
        #         step=completed_steps,
        #     )

        if eval_loss < best_loss:
            best_loss = eval_loss
            best_model_dir = f"epoch_{epoch}"

            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        # if args.push_to_hub and epoch < args.num_train_epochs - 1:
        #     accelerator.wait_for_everyone()
        #     unwrapped_model = accelerator.unwrap_model(model)
        #     unwrapped_model.save_pretrained(
        #         args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        #     )
        #     if accelerator.is_main_process:
        #         tokenizer.save_pretrained(args.output_dir)
        #         repo.push_to_hub(
        #             commit_message=f"Training in progress epoch {epoch}", blocking=False, auto_lfs_prune=True
        #         )

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir, safe_serialization=False)
        
        if accelerator.is_main_process:
            # Get a list of all directories in the parent directory
            all_dirs_exclude_best = [str(pth) for pth in Path(args.output_dir).iterdir() if pth.is_dir()]# and os.path.basename(pth) != best_model_dir]

            # Sort the directories by modification time (latest first)
            all_dirs_exclude_best.sort(key=os.path.getmtime, reverse=True)

            if len(all_dirs_exclude_best) > args.max_save_limit-1:
                dirs_to_keep = all_dirs_exclude_best[:args.max_save_limit-1]

                # Loop through all directories and remove them if they are not in the list of directories to keep
                for dir in all_dirs_exclude_best:
                    if dir not in dirs_to_keep:
                        shutil.rmtree(dir)
        
        if args.early_stopping_patience is not None and epochs_no_improve > args.early_stopping_patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break

    accelerator.wait_for_everyone()
    logger.info("*** Test ***")
    logger.info("Loading best checkpoint from: %s", os.path.join(args.output_dir, best_model_dir))
    accelerator.load_state(os.path.join(args.output_dir, best_model_dir))
    model.eval()
    encoder.eval()
    accelerator.wait_for_everyone()

    mlm_preds = []
    mlm_labels = []
    secondary_logits = []
    log_labels = []

    for steps, batch in enumerate(test_dataloader):
        with torch.no_grad():
            encodings = encoder(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            outputs = model(encodings.last_hidden_state, labels=batch['labels'])

        mlm_preds.append(accelerator.gather(precompute_logits(outputs.logits, k=5)).cpu().numpy())
        mlm_labels.append(accelerator.gather(batch["labels"]).cpu().numpy())
        secondary_logits.append(accelerator.gather(outputs.secondary_logits).cpu().numpy())
        log_labels.append(accelerator.gather(batch["log_labels"]).cpu().numpy())

    mlm_preds = np.concatenate(mlm_preds)
    mlm_labels = np.concatenate(mlm_labels)
    secondary_logits = np.concatenate(secondary_logits)
    log_labels = np.concatenate(log_labels)
    results = compute_metrics_for_test(mlm_preds, mlm_labels, secondary_logits, log_labels)
    logger.info(f"{results}")

    if args.with_tracking:
        accelerator.end_training()

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save, safe_serialization=False
        )
        # if accelerator.is_main_process:
        #     tokenizer.save_pretrained(args.output_dir)
        #     if args.push_to_hub:
        #         repo.push_to_hub(commit_message="End of training", auto_lfs_prune=True)

        with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
            # json.dump({"perplexity": perplexity}, f)
            json.dump(results, f)


if __name__ == "__main__":
    main()
