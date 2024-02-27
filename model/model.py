from transformers.configuration_utils import PretrainedConfig
from transformers.models.roberta.modeling_roberta import RobertaConfig, RobertaPreTrainedModel, RobertaModel, RobertaLMHead, RobertaClassificationHead
from transformers.modeling_outputs import MaskedLMOutput, SequenceClassifierOutput
from dataclasses import dataclass
from transformers.modeling_outputs import ModelOutput
from info_nce import InfoNCE, info_nce
import torch.distributed as dist

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss

from typing import Optional, Tuple, Union

@dataclass
class LogAnomalyRobertOutput(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    mlm_logits: Optional[torch.FloatTensor] = None
    # cls_logits: Optional[torch.FloatTensor] = None
    # dist: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class LogRobertaConfig(RobertaConfig):
    def __init__(
        self,
        # num_of_components=10,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # self.num_of_components = num_of_components

# class CondensedLinear(nn.Module):
#     def __init__(self, in_features: int, out_features: int, n_component: int, bias: bool = True, dropout: float = 0.1):
#         super(CondensedLinear, self).__init__()
#         self.lora_A = nn.Linear(in_features, n_component, bias=bias)
#         self.lora_B = nn.Linear(n_component*2, n_component, bias=bias)
#         self.lora_C = nn.Linear(n_component, out_features, bias=bias)
#         self.dropout = nn.Dropout(dropout)


#     def forward(self, x1, x2):
#         x1 = self.lora_A(x1)
#         x2 = self.lora_A(x2)
#         x = torch.cat((x1, x2), dim=-1)
#         x = torch.tanh(self.lora_B(x))
#         x = self.dropout(x)
#         x = torch.tanh(self.lora_C(x))
#         return self.dropout(x)
        
class AE(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        # Building an linear encoder with Linear
        self.hidden_size = hidden_size
        self.encoder = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size//2),
            nn.Tanh(),
            nn.Linear(self.hidden_size//2, self.hidden_size//4),
            nn.Tanh(),
            nn.Linear(self.hidden_size//4, self.hidden_size//8),
            nn.Tanh(),
            nn.Linear(self.hidden_size//8, self.hidden_size//16),
            nn.Tanh(),
            nn.Linear(self.hidden_size//16, self.hidden_size//32),
            nn.Tanh(),
            nn.Linear(self.hidden_size//32, self.hidden_size//64)
        )
         
        # Building an linear decoder with Linear
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_size//64, self.hidden_size//32),
            nn.Tanh(),
            nn.Linear(self.hidden_size//32, self.hidden_size//16),
            nn.Tanh(),
            nn.Linear(self.hidden_size//16, self.hidden_size//8),
            nn.Tanh(),
            nn.Linear(self.hidden_size//8, self.hidden_size//4),
            nn.Tanh(),
            nn.Linear(self.hidden_size//4, self.hidden_size//2),
            nn.Tanh(),
            nn.Linear(self.hidden_size//2, self.hidden_size)
        )
 
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class VAE(nn.Module):

    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.latent_size = hidden_size//64
        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size//2),
            nn.Tanh(),
            nn.Linear(self.hidden_size//2, self.hidden_size//4),
            nn.Tanh(),
            nn.Linear(self.hidden_size//4, self.hidden_size//8),
            nn.Tanh(),
            nn.Linear(self.hidden_size//8, self.hidden_size//16),
            nn.Tanh(),
            nn.Linear(self.hidden_size//16, self.hidden_size//32),
            nn.Tanh(),
            nn.Linear(self.hidden_size//32, self.hidden_size//64)
            )
        
        # latent mean and variance 
        self.mean_layer = nn.Linear(self.latent_size, 2)
        self.logvar_layer = nn.Linear(self.latent_size, 2)
        
        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(2, self.hidden_size//64),
            nn.Tanh(),
            nn.Linear(self.hidden_size//64, self.hidden_size//32),
            nn.Tanh(),
            nn.Linear(self.hidden_size//32, self.hidden_size//16),
            nn.Tanh(),
            nn.Linear(self.hidden_size//16, self.hidden_size//8),
            nn.Tanh(),
            nn.Linear(self.hidden_size//8, self.hidden_size//4),
            nn.Tanh(),
            nn.Linear(self.hidden_size//4, self.hidden_size//2),
            nn.Tanh(),
            nn.Linear(self.hidden_size//2, self.hidden_size),
            )
     
    def encode(self, x):
        x = self.encoder(x)
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        return mean, logvar

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var)      
        z = mean + var*epsilon
        return z

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterization(mean, logvar)
        x_hat = self.decode(z)
        return x_hat, mean, logvar
    
class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp
    
class LogAnomalyRobertaClsHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        return x

class LogAnomalyRobertaModel(RobertaPreTrainedModel):
    def __init__(self, config, center=None):
        super().__init__(config)

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        # self.lora = CondensedLinear(config.hidden_size, config.hidden_size, config.num_of_components, dropout=config.hidden_dropout_prob)
        self.ae = AE(config.hidden_size)
        # self.center = None
        self.sim = Similarity(0.1)
        self.lm_head = RobertaLMHead(config)
        self.cls_head = LogAnomalyRobertaClsHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.roberta.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.roberta.embeddings.word_embeddings = value
    
    # def vae_loss_function(self, x, x_hat, mean, log_var):
    #     reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    #     KLD = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

    #     return reproduction_loss + KLD
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        # secondary_input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        # secondary_attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        # secondary_token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        # secondary_position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        # secondary_head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        # secondary_inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        log_labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        
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

        # with torch.no_grad():
        #     secondary_outputs = self.roberta(
        #         secondary_input_ids,
        #         attention_mask=secondary_attention_mask,
        #         token_type_ids=secondary_token_type_ids,
        #         position_ids=secondary_position_ids,
        #         head_mask=secondary_head_mask,
        #         inputs_embeds=secondary_inputs_embeds,
        #         encoder_hidden_states=encoder_hidden_states,
        #         encoder_attention_mask=encoder_attention_mask,
        #         output_attentions=output_attentions,
        #         output_hidden_states=output_hidden_states,
        #         return_dict=return_dict,
        #     )
        # secondary_sequence_output = secondary_outputs[0]

        # enhanced_sequence_output = self.lora(sequence_output, secondary_sequence_output)
        sequence_output_reconstructed = self.ae(sequence_output)
        prediction_scores = self.lm_head(sequence_output)
        z1 = self.cls_head(sequence_output)
        z2 = self.cls_head(sequence_output_reconstructed)

        total_loss = None

        masked_lm_loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(prediction_scores.device)
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            total_loss = masked_lm_loss 

        ae_loss_fct = MSELoss()
        ae_loss = ae_loss_fct(z2, z1)
        total_loss = ae_loss if total_loss is None else total_loss + ae_loss
        # vae_loss = self.vae_loss_function(z1, z2, mean, logvar)
        # total_loss = vae_loss if total_loss is None else total_loss + vae_loss
        # hypersphere_loss = None
        # dist = None
        # if self.center is not None:
        #     loss_fct = MSELoss()
        #     hypersphere_loss = loss_fct(cls_logits.squeeze(), self.center.expand(input_ids.shape[0], -1))
        #     total_loss = hypersphere_loss

        #     dist = torch.sum((cls_logits - self.center) ** 2, dim=1)
        # z1_list = [torch.zeros_like(z1) for _ in range(dist.get_world_size())]
        # z2_list = [torch.zeros_like(z2) for _ in range(dist.get_world_size())]
        # # Allgather
        # dist.all_gather(tensor_list=z1_list, tensor=z1.contiguous())
        # dist.all_gather(tensor_list=z2_list, tensor=z2.contiguous())

        # # Since allgather results do not have gradients, we replace the
        # # current process's corresponding embeddings with original tensors
        # z1_list[dist.get_rank()] = z1
        # z2_list[dist.get_rank()] = z2
        # # Get full batch embeddings: (bs x N, hidden)
        # z1 = torch.cat(z1_list, 0)
        # z2 = torch.cat(z2_list, 0)

        # cos_sim = self.sim(z1.unsqueeze(1), z2.unsqueeze(0))

        # sim_cse_labels = torch.arange(cos_sim.size(0)).long().to(cos_sim.device)
        # sim_cse_loss_fct = nn.CrossEntropyLoss()
        # sim_cse_loss = sim_cse_loss_fct(cos_sim, sim_cse_labels)
        # total_loss = sim_cse_loss if total_loss is None else total_loss + sim_cse_loss

        # ctr_loss_fct = InfoNCE()
        # ctr_loss = ctr_loss_fct(z1, z2)
        # total_loss = ctr_loss if total_loss is None else total_loss + ctr_loss

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output
        

        return LogAnomalyRobertOutput(
            loss=total_loss,
            mlm_logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )