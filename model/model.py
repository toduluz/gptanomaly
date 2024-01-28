from transformers.models.roberta.modeling_roberta import RobertaConfig, RobertaPreTrainedModel, RobertaModel, RobertaLMHead
from transformers.modeling_outputs import MaskedLMOutput

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from typing import Optional, Tuple, Union


class LogRobertaConfig(RobertaConfig):
    def __init__(
        self,
        num_of_components=10,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_of_components = num_of_components

class CondensedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, n_component: int, bias: bool = True, dropout: float = 0.1):
        super(CondensedLinear, self).__init__()
        self.lora_A = nn.Linear(in_features, n_component, bias=False)
        self.lora_B = nn.Linear(n_component, out_features, bias=bias)
        self.dropout = nn.Dropout(dropout)


    def forward(self, input):
        x = self.lora_A(input)
        x = torch.tanh(self.lora_B(x))
        return self.dropout(x)

class LogRobertaModelForMaskedLM(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.lm_head = RobertaLMHead(config)
        self.lora = CondensedLinear(config.hidden_size, config.hidden_size, config.num_of_components, dropout=config.hidden_dropout_prob)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        secondary_input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        secondary_attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        secondary_token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        log_labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        kwargs (`Dict[str, any]`, optional, defaults to *{}*):
            Used to hide legacy arguments that have been deprecated.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]

        with torch.no_grad():
            secondary_outputs = self.roberta(
                secondary_input_ids,
                attention_mask=secondary_attention_mask,
                token_type_ids=secondary_token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        secondary_sequence_output = secondary_outputs[0]
        secondary_enhanced = self.lora(secondary_sequence_output)

        sequence_output = sequence_output + secondary_enhanced

        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(prediction_scores.device)
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )