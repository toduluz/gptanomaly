from arguments import InitializationArguments

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser

# from models import AnomalyGpt


# Configuration
parser = HfArgumentParser(InitializationArguments)
args = parser.parse_args()

# Load codeparrot tokenizer trained for Python code tokenization
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, model_max_length=args.model_max_length)

# Config: "scale_attn_by_layer_idx" and "reorder_and_upcast_attn" are Mistral stability tweaks
config_kwargs = {
    "vocab_size": len(tokenizer),
    "scale_attn_by_inverse_layer_idx": True,
    "reorder_and_upcast_attn": True,
    "n_positions": args.model_max_length,
    "n_ctx": args.model_max_length,
    "num_labels": 2
}

# Load model config (GPT-2 large in this case)
config = AutoConfig.from_pretrained(args.config_name, **config_kwargs)
print(config)
# Initialize new model with config
model = AutoModelForSequenceClassification.from_pretrained(args.config_name, config=config)
# model = AnomalyGpt(config)

# Save model to the hub
model.save_pretrained("outputs/model")
