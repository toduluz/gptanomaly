from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainingArguments:
    """
    Configuration for training model.
    """

    model_ckpt: Optional[str] = field(
        default="codeparrot/codeparrot", metadata={"help": "Model name or path of model to be trained."}
    )
    save_dir: Optional[str] = field(
        default="./", metadata={"help": "Save dir where model repo is cloned and models updates are saved to."}
    )
    dataset_name_train: Optional[str] = field(
        default="codeparrot/codeparrot-clean-train", metadata={"help": "Name or path of training dataset."}
    )
    dataset_name_valid: Optional[str] = field(
        default="codeparrot/codeparrot-clean-valid", metadata={"help": "Name or path of validation dataset."}
    )
    train_batch_size: Optional[int] = field(default=2, metadata={"help": "Batch size for training."})
    valid_batch_size: Optional[int] = field(default=2, metadata={"help": "Batch size for evaluation."})
    weight_decay: Optional[float] = field(default=0.1, metadata={"help": "Value of weight decay."})
    shuffle_buffer: Optional[int] = field(
        default=10000, metadata={"help": "Size of buffer used to shuffle streaming dataset."}
    )
    learning_rate: Optional[float] = field(default=2e-4, metadata={"help": "Learning rate fo training."})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "Learning rate."})
    num_warmup_steps: Optional[int] = field(
        default=750, metadata={"help": "Number of warmup steps in the learning rate schedule."}
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=16, metadata={"help": "Number of gradient accumulation steps."}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "Use gradient checkpointing to reduce memory footprint."}
    )
    max_train_steps: Optional[int] = field(default=50000, metadata={"help": "Maximum number of training steps."})
    max_eval_steps: Optional[int] = field(
        default=-1, metadata={"help": "Maximum number of evaluation steps. If -1 the full dataset is evaluated."}
    )
    seq_length: Optional[int] = field(default=1024, metadata={"help": "Sequence lengths used for training."})
    seed: Optional[int] = field(default=1, metadata={"help": "Training seed."})
    save_checkpoint_steps: Optional[int] = field(
        default=1024,
        metadata={"help": "Interval to save checkpoints. Measured as number of forward passes not training steps."},
    )
    resume_from_checkpoint: Optional[str] = field(
        default=None, metadata={"help": "States path if the training should continue from a checkpoint folder."}
    )
    tokenized: Optional[bool] = field(default=False, metadata={"help": "If True the data is pretokenized."})


@dataclass
class EvaluationArguments:
    """
    Configuration for evaluating model.
    """

    model_ckpt: Optional[str] = field(
        default="codeparrot/codeparrot", metadata={"help": "Model name or path of model to be evaluated."}
    )
    dataset_name: Optional[str] = field(
        default="codeparrot/codeparrot-clean-valid", metadata={"help": "Name or path of validation dataset."}
    )
    batch_size: Optional[int] = field(default=2, metadata={"help": "Batch size used for evaluation."})
    max_eval_steps: Optional[int] = field(
        default=-1, metadata={"help": "Maximum number of evaluation steps. If -1 the full dataset is evaluated."}
    )
    seq_length: Optional[int] = field(default=1024, metadata={"help": "Length of sequences to be evaluated."})
    seed: Optional[int] = field(default=1, metadata={"help": "Random seed used for evaluation."})

@dataclass
class TokenizerTrainingArguments:
    """
    Configuration for tokenizer training.
    """

    base_tokenizer: Optional[str] = field(
        default="gpt2", metadata={"help": "Base tokenizer to build new tokenizer from."}
    )
    data_dir: Optional[str] = field(
        default="data", metadata={"help": "Data directory to train tokenizer on."}
    )
    dataset_name: Optional[str] = field(
        default="hdfs", metadata={"help": "Dataset name to train tokenizer on."}
    )
    text_column: Optional[str] = field(default="text", metadata={"help": "Column containing text data to process."})
    vocab_size: Optional[int] = field(default=50257, metadata={"help": "Vocab size"})
    n_examples: Optional[int] = field(
        default=50000, metadata={"help": "Number of examples to train the tokenizer on."}
    )
    model_max_length: Optional[int] = field(default=1024, metadata={"help": "Model max length"})
    tokenizer_name: Optional[str] = field(default="anomalygpt", metadata={"help": "Name of new tokenizer."})
    # push_to_hub: Optional[bool] = field(default=True, metadata={"help": "Push saved tokenizer to the hub."})


@dataclass
class PretokenizationArguments:
    """
    Configuration for data pretokenization.
    """

    tokenizer_dir: Optional[str] = field(
        default="outputs", metadata={"help": "Name or path to the tokenizer."}
    )
    data_dir: Optional[str] = field(
        default="data", metadata={"help": "Data directory."}
    )
    dataset_name: Optional[str] = field(
        default="hdfs", metadata={"help": "Name to the dataset to pretokenize."}
    )
    model_max_length: Optional[int] = field(default=1024, metadata={"help": "Model max length"})
    # tokenized_data_repo: Optional[str] = field(
    #     default="tokenized-codeparrot-train", metadata={"help": "Repo name of the pretokenized data."}
    # )
    num_workers: Optional[int] = field(default=None, metadata={"help": "Number of workers used for code evaluation."})


@dataclass
class InitializationArguments:
    """
    Configuration for initializing new model.
    """

    config_name: Optional[str] = field(
        default="gpt2-large", metadata={"help": "Configuration to use for model initialization."}
    )
    tokenizer_name: Optional[str] = field(
        default="gpt2-large", metadata={"help": "Tokenizer attached to model."}
    )
    model_max_length: Optional[int] = field(default=1024, metadata={"help": "Model max length"})
    # model_name: Optional[str] = field(default="codeparrot", metadata={"help": "Name of the created model."})
    # push_to_hub: Optional[bool] = field(default=True, metadata={"help": "Push saved tokenizer to the hub."})
