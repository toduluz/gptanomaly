from transformers.models.roberta.modeling_roberta import RobertaConfig, RobertaPreTrainedModel, RobertaLMHead, RobertaEmbeddings
from transformers.modeling_outputs import MaskedLMOutput
from dataclasses import dataclass
from transformers.modeling_outputs import ModelOutput

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from typing import Optional, Tuple, Union

@dataclass
class LogRobertAEOutput(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    secondary_logits: torch.FloatTensor = None

class LogRobertaAEConfig(RobertaConfig):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

class LanguageModelAutoencoder(nn.Module):
    def __init__(self, hidden_size, latent_size):
        super().__init__()

        # Map the encoder output to the latent space
        self.latent = nn.Linear(hidden_size, latent_size)

        # Map the latent representation back to the hidden size
        self.expand = nn.Linear(latent_size, hidden_size)

    def forward(self, x):
        # Compute the latent representation
        latent_representation = self.latent(x)

        # Expand the latent representation
        expanded_representation = self.expand(latent_representation)

        return expanded_representation

class LogRobertaAEModelForMaskedLM(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.lm_head = RobertaLMHead(config)
        self.ae = LanguageModelAutoencoder(config.hidden_size, config.hidden_size//4)

        self.embeddings = RobertaEmbeddings(config)


        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    def forward(self, x, labels=None):
    
        ae_output = self.ae(x)
        prediction_scores = self.lm_head(ae_output)

        # Compute the autoencoder loss
        loss_fct_ae = MSELoss()
        ae_loss = loss_fct_ae(ae_output.reshape(-1, self.config.hidden_size), x.reshape(-1, self.config.hidden_size))

        masked_lm_loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(prediction_scores.device)
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        
        if masked_lm_loss is None:
            total_loss = ae_loss
        else:
            total_loss = ae_loss + masked_lm_loss

        return LogRobertAEOutput(
            loss=total_loss,
            logits=prediction_scores,
            secondary_logits=ae_output[:, 0, :].reshape(-1, self.config.hidden_size)
        )