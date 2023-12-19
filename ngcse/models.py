# import logging
import re
import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from functools import reduce

from transformers.utils import logging
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel
from transformers.models.roberta.configuration_roberta import RobertaConfig
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel
from transformers.models.bert.configuration_bert import BertConfig

from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions
from typing import Union, Dict, Any, Tuple, Union

logger = logging.get_logger()

class NGBertConfig(BertConfig):
    def __init__(self,
        use_native_loss: bool = True,
        use_loss3: bool = False,
        use_loss4: bool = False,
        temp: float = 5e-2, 
        margins: list = [[5e-3, 5e-2], []],
        beta: float = 1.,
        gamma: float = 2e-9, 
        pooler_type: str = 'cls',
        feature_size: int = 768,
        num_layers: int = 12,
        optional: str = '',
        **kwargs
    ): 
        super().__init__(**kwargs)
        self.pooler_type = pooler_type
        self.feature_size = feature_size
        self.temp = temp
        self.margins = margins
        self.beta = beta
        self.gamma = gamma
        self.num_layers = num_layers

        self.use_native_loss = use_native_loss
        self.use_loss3 = use_loss3
        self.use_loss4 = use_loss4

        self.optional = optional

class NGRobertaConfig(RobertaConfig):
    def __init__(self,
        use_native_loss: bool = True,
        use_loss3: bool = False,
        use_loss4: bool = False,
        temp: float = 5e-2, 
        margins: list = [[5e-3, 5e-2], []],
        beta: float = 1.,
        gamma: float = 2e-9, 
        pooler_type: str = 'cls',
        feature_size: int = 768,
        num_layers: int = 12,
        optional: str = '',
        **kwargs
    ): 
        super().__init__(**kwargs)
        self.pooler_type = pooler_type
        self.feature_size = feature_size
        self.temp = temp
        self.margins = margins
        self.beta = beta
        self.gamma = gamma
        self.num_layers = num_layers

        self.use_native_loss = use_native_loss
        self.use_loss3 = use_loss3
        self.use_loss4 = use_loss4

        self.optional = optional

class NGBert(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = ['pooler', 'cls']
    config_class = NGBertConfig

    def __init__(self, config, *model_args, **model_kwargs):
        super().__init__(config)
        # self.model_args = model_kwargs["model_args"]
        self.bert = BertModel(config, add_pooling_layer=False)

        self.prompt = [
            {
                'input_ids': [torch.tensor([2023, 6251, 1024, 1000]), torch.tensor([1000, 2965, 103])],
                'token_type_ids': [torch.tensor([0, 0, 0, 0]), torch.tensor([0, 0, 0])],
                'attention_mask': [torch.tensor([1, 1, 1, 1]), torch.tensor([1, 1, 1])]
            },
            {
                'input_ids': [torch.tensor([2023, 6251, 1997, 1000]), torch.tensor([1000, 2965, 103])],
                'token_type_ids': [torch.tensor([0, 0, 0, 0]), torch.tensor([0, 0, 0])],
                'attention_mask': [torch.tensor([1, 1, 1, 1]), torch.tensor([1, 1, 1])]
            }
        ]

        self.mlp = nn.Linear(config.hidden_size, config.feature_size)
        self.similarity = lambda x,y: F.cosine_similarity(x, y, dim=-1)

        self.named_parameters_cache = {k: v.detach().clone() for k, v in self.named_parameters() 
                                       if 'bert' in k and v.requires_grad == True}

        self.loss_logs = {'loss_all': [], 'loss1': [], 'loss2': [], 'loss3': [], 'loss4': []}

    def add_prompt(self, input_ids, token_type_ids, attention_mask):
        def _add_prompt(prompt, input_ids, token_type_ids, attention_mask):
            sent_len = attention_mask.sum(dim=-1)
            n_input_ids = []
            if token_type_ids is not None:
                n_token_type_ids = []
            n_attention_mask = []
            for s_idx in range(input_ids.shape[0]):
                n_input_ids.append(torch.cat([input_ids[s_idx][:1], prompt['input_ids'][0].to(input_ids[s_idx][:1]), 
                    input_ids[s_idx][1:sent_len[s_idx] - 1], prompt['input_ids'][1].to(input_ids[s_idx][:1]), 
                    input_ids[s_idx][sent_len[s_idx] - 1: ]], dim=0))
                if token_type_ids is not None:
                    n_token_type_ids.append(torch.cat([token_type_ids[s_idx][:1], prompt['token_type_ids'][0].to(token_type_ids[s_idx][:1]), 
                        token_type_ids[s_idx][1:sent_len[s_idx] - 1], prompt['token_type_ids'][1].to(token_type_ids[s_idx][:1]), 
                        token_type_ids[s_idx][sent_len[s_idx] - 1: ]], dim=0))
                n_attention_mask.append(torch.cat([attention_mask[s_idx][:1], prompt['attention_mask'][0].to(attention_mask[s_idx][:1]),
                    attention_mask[s_idx][1:sent_len[s_idx] - 1], prompt['attention_mask'][1].to(attention_mask[s_idx][:1]), 
                    attention_mask[s_idx][sent_len[s_idx] - 1: ]], dim=0))
            n_input_ids = torch.stack(n_input_ids, dim=0)
            if token_type_ids is not None:
                n_token_type_ids = torch.stack(n_token_type_ids, dim=0)
            else:
                n_token_type_ids = None
            n_attention_mask = torch.stack(n_attention_mask, dim=0)
            return n_input_ids, n_token_type_ids, n_attention_mask
        
        if isinstance(input_ids, list):

            if input_ids[0].shape[0] > 0:
                n_input_ids0 = []
                if token_type_ids[0] is not None:
                    n_token_type_ids0 = []
                n_attention_mask0 = []
                for idx1 in range(input_ids[0].shape[1]):
                    r_input_ids, r_token_type_ids, r_attention_mask = _add_prompt(
                        self.prompt[0], input_ids[0][:, idx1], token_type_ids[0][:, idx1] if token_type_ids[0] is not None else None,
                        attention_mask[0][:, idx1]
                    )
                    n_input_ids0.append(r_input_ids)
                    if token_type_ids[0] is not None:
                        n_token_type_ids0.append(r_token_type_ids)
                    n_attention_mask0.append(r_attention_mask)
                input_ids[0] = torch.stack(n_input_ids0, dim=1)
                if token_type_ids[0] is not None:
                    token_type_ids[0] = torch.stack(n_token_type_ids0, dim=1)
                else:
                    token_type_ids[0] = None
                attention_mask[0] = torch.stack(n_attention_mask0, dim=1)

            if input_ids[1].shape[0] > 0:
                n_input_ids1 = []
                if token_type_ids[1] is not None:
                    n_token_type_ids1 = []
                n_attention_mask1 = []
                for idx1 in range(input_ids[1].shape[1]):
                    if input_ids[1].shape[1] == 2:
                        r_input_ids, r_token_type_ids, r_attention_mask = _add_prompt(
                            self.prompt[idx1], input_ids[1][:, idx1], token_type_ids[1][:, idx1] if token_type_ids[1] is not None else None,
                            attention_mask[1][:, idx1]
                        )
                    else:
                        r_input_ids, r_token_type_ids, r_attention_mask = _add_prompt(
                            self.prompt[0], input_ids[1][:, idx1], token_type_ids[1][:, idx1] if token_type_ids[1] is not None else None,
                            attention_mask[1][:, idx1]
                        )
                    n_input_ids1.append(r_input_ids)
                    if token_type_ids[1] is not None:
                        n_token_type_ids1.append(r_token_type_ids)
                    n_attention_mask1.append(r_attention_mask)
                input_ids[1] = torch.stack(n_input_ids1, dim=1)
                if token_type_ids[1] is not None:
                    token_type_ids[1] = torch.stack(n_token_type_ids1, dim=1)
                else:
                    token_type_ids[1] = None
                attention_mask[1] = torch.stack(n_attention_mask1, dim=1)
            
            if input_ids[0].shape[-1] != input_ids[1].shape[-1]:
                if input_ids[0].shape[0] == 0:
                    input_ids[0] = input_ids[0].reshape(*(input_ids[0].shape[:-1] + input_ids[1].shape[-1:]))
                    if token_type_ids[0] is not None:
                        token_type_ids[0] = token_type_ids[0].reshape(*(token_type_ids[0].shape[:-1] + token_type_ids[1].shape[-1:]))
                    attention_mask[0] = attention_mask[0].reshape(*(attention_mask[0].shape[:-1] + attention_mask[1].shape[-1:]))
                elif input_ids[1].shape[0] == 0:
                    input_ids[1] = input_ids[1].reshape(*(input_ids[1].shape[:-1] + input_ids[0].shape[-1:]))
                    if token_type_ids[1] is not None:
                        token_type_ids[1] = token_type_ids[1].reshape(*(token_type_ids[1].shape[:-1] + token_type_ids[0].shape[-1:]))
                    attention_mask[1] = attention_mask[1].reshape(*(attention_mask[1].shape[:-1] + attention_mask[0].shape[-1:]))
                else:
                    raise Exception

        else:

            if len(input_ids.shape) == 3:
                n_input_ids = []
                if token_type_ids is not None:
                    n_token_type_ids = []
                n_attention_mask = []

                for idx1 in range(input_ids.shape[1]):
                    r_input_ids, r_token_type_ids, r_attention_mask = _add_prompt(
                        self.prompt[idx1 % 2], input_ids[:, idx1], token_type_ids[:, idx1] if token_type_ids is not None else None, 
                        attention_mask[:, idx1]
                    )
                    n_input_ids.append(r_input_ids)
                    if token_type_ids is not None:
                        n_token_type_ids.append(r_token_type_ids)
                    n_attention_mask.append(r_attention_mask)
                input_ids = torch.stack(n_input_ids, dim=1)
                if token_type_ids is not None:
                    token_type_ids = torch.stack(n_token_type_ids, dim=1)
                attention_mask = torch.stack(n_attention_mask, dim=1)

            else:
                input_ids, token_type_ids, attention_mask = _add_prompt(
                    self.prompt[0], input_ids, token_type_ids, attention_mask
                )

        return input_ids, token_type_ids, attention_mask

    def display_loss(self, mode: Union[int, str]):
        if dist.is_initialized() and dist.get_rank() != 0:
            return
        if mode == 'avg':
            logger.info("Average loss:\n{}".format(
                {k: sum(v) / len(v) for k, v in self.loss_logs.items() if v}
            ))
        elif isinstance(mode, int):
            logger.info("Loss every {} inputs:\n{}".format(
                mode, {
                    k: [(i_loss, sum(v[: i_index + 1]) / (i_index + 1)) for i_index, i_loss in enumerate(v) if (i_index % mode) == 0] 
                    for k, v in self.loss_logs.items() if v
                }
            ))
        else:
            logger.error(f"Can not display loss in mode {mode}")

    def _get_embedding(
        self, attention_mask: torch.Tensor, outputs: Dict[str, torch.Tensor],
        with_mlp: bool = True
    ) -> torch.Tensor:
        last_hidden: torch.Tensor = outputs.last_hidden_state # (bs, seq_len, hidden_len)
        hidden_states: torch.Tensor = outputs.hidden_states  # Tuple of (bs, seq_len, hidden_len)

        if 'cls' in self.config.pooler_type:
            cls_embedding = last_hidden[:, 0]
            if with_mlp:
                return self.mlp(cls_embedding) #(bs, hidden_len)
            else:
                return cls_embedding
        elif 'avg' in self.config.pooler_type:
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / 
                     attention_mask.sum(-1).unsqueeze(-1)) #(bs, hidden_len)
        elif 'mask' in self.config.pooler_type:
            mask_embedding = []
            sent_len = attention_mask.sum(dim=-1) 
            for idx in range(last_hidden.shape[0]):
                mask_embedding.append(last_hidden[idx][sent_len[idx] - 2])
            mask_embedding = torch.stack(mask_embedding, dim=0)
            if with_mlp:
                return self.mlp(mask_embedding) #(bs, hidden_len)
            else:
                return mask_embedding
        else:
            raise NotImplementedError

    def _ng_forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        loss_pair=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None
    ) -> SequenceClassifierOutput:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        ori_input_ids = input_ids


        aigen_batch_size = input_ids[0].size(0)
        aigen_sent_num = input_ids[0].size(1)
        other_batch_size = input_ids[1].size(0)
        other_sent_num = input_ids[1].size(1)

        
        inp_input_ids = torch.cat([input_ids[0].reshape(-1, input_ids[0].shape[-1]), 
                                       input_ids[1].reshape(-1, input_ids[1].shape[-1])], dim=0) # shape of [abs * aigen_sent_num + obs * 2, seq_len]
        inp_attention_mask = torch.cat([attention_mask[0].reshape(-1, attention_mask[0].shape[-1]),
                                        attention_mask[1].reshape(-1, attention_mask[1].shape[-1])], dim=0)
        if token_type_ids is not None and token_type_ids[1] is not None:
            inp_token_type_ids = torch.cat([token_type_ids[0].reshape(-1, token_type_ids[0].shape[-1]),
                                            token_type_ids[1].reshape(-1, token_type_ids[1].shape[-1])], dim=0)
        else:
            inp_token_type_ids = None

        # Get raw embeddings
        outputs = self.bert(
            inp_input_ids,
            attention_mask=inp_attention_mask,
            token_type_ids=inp_token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True, # if cls.config.pooler_type in ['avg_top2', 'avg_first_last'] else False,
            return_dict=True,
        )

        embedding = self._get_embedding(inp_attention_mask, outputs, with_mlp=True) # (abs * aigen_sent_num + obs * 2, hidden_size)

        if dist.is_initialized() and self.training:
            raise NotImplementedError # FIXME unavailable when abs and obs are not fixed between batches.

        aigen_h_embedding = embedding[:aigen_batch_size * aigen_sent_num].reshape(aigen_batch_size, aigen_sent_num, embedding.shape[-1])
        other_h_embedding = embedding[aigen_batch_size * aigen_sent_num:].reshape(other_batch_size, other_sent_num, embedding.shape[-1])
        
        temp, margins = self.config.temp, self.config.margins
        use_loss4 = self.config.use_loss4
        use_loss3 = self.config.use_loss3
        use_native_loss = self.config.use_native_loss
        optional = self.config.optional.split(',')

        if use_native_loss:
            
            loss1_fn = nn.CrossEntropyLoss()

            aigen_pos_idx = 2 if aigen_sent_num > 3 else 1
            embeds_0 = torch.cat(
                ([other_h_embedding[:, :1]] if other_sent_num else []) + 
                ([aigen_h_embedding[:, :1]] if aigen_sent_num else []),
                dim=0
            )
            embeds_1 = torch.cat(
                ([other_h_embedding[None, :, 1]] if other_sent_num else []) +
                ([aigen_h_embedding[None, :, aigen_pos_idx]] if aigen_sent_num else []) +
                ([other_h_embedding[None, :, -1]] if other_sent_num > 2 else []) +
                ([aigen_h_embedding[None, :, -1]] if aigen_sent_num > 2 and not use_loss3 else []),
                dim=1
            )

            similarity = self.similarity(embeds_0, embeds_1)
            
            labels = torch.arange(similarity.shape[0]).to(dtype=torch.long, device=self.device)
            loss1 = loss1_fn(similarity / temp, labels)

            loss2 = None

        else:

            if other_batch_size > 0:
                # loss1: CrossEntropy or InforNCE
                loss1_fn = nn.CrossEntropyLoss()
                embeds_0 = other_h_embedding[:, :1] # (obs, 1, hidden_size)
                embeds_1 = torch.cat([other_h_embedding[:, 1], 
                                      aigen_h_embedding[:, 1]] + 
                                     ([] if other_sent_num == 2 else [other_h_embedding[:, -1]])
                                    , dim=0).unsqueeze(0) # (1, obs + abs * asn, hidden_size)
                # embeds_1 = other_h_embedding[None, :, 1] # alternative
                similarity = self.similarity(embeds_0, embeds_1) # (obs, obs + abs * asn)
                labels = torch.arange(similarity.shape[0]).to(dtype=torch.long, device=self.device)
                loss1 = loss1_fn(similarity / temp, labels)
            else:
                loss1 = None
            
            
            if aigen_batch_size > 0:
                # loss2: CrossEntropy or InforNCE
                loss2_fn = nn.CrossEntropyLoss()
                loss2 = torch.tensor(0., device=self.device)
                aigen_pos_sent_num = 2 # FIXME
                embeds_0 = aigen_h_embedding[:, :1] # (abs, 1, hidden_size)
                
                for aigen_pos_idx in range(1, aigen_pos_sent_num + 1):
                    if aigen_pos_idx == aigen_pos_sent_num:
                        embeds_1 = torch.cat([aigen_h_embedding[:, aigen_pos_idx], aigen_h_embedding[:, -1], other_h_embedding[:, 1]], dim=0).unsqueeze(0)
                    else:
                        embeds_1 = torch.cat([aigen_h_embedding[:, aigen_pos_idx], other_h_embedding[:, 1]], dim=0).unsqueeze(0) # (1, abs + obs, hidden_size)
                    # w/o hn
                    # embeds_1 = torch.cat([aigen_h_embedding[:, aigen_pos_idx], other_h_embedding[:, 1]], dim=0).unsqueeze(0) # (1, abs + obs, hidden_size)
                    # w hn: -1
                    # embeds_1 = torch.cat([aigen_h_embedding[:, aigen_pos_idx], aigen_h_embedding[:, -1], other_h_embedding[:, 1]], dim=0).unsqueeze(0) # (1, 2 * abs + obs, hidden_size)
                    
                    similarity = self.similarity(embeds_0, embeds_1) # (abs, obs + abs)
                    labels = torch.arange(similarity.shape[0]).to(dtype=torch.long, device=self.device)
                    loss2 += loss2_fn(similarity / temp, labels)
                loss2 /= aigen_pos_sent_num
    
            else:
                loss2 = None

        if use_loss3 and aigen_batch_size > 0:
            # loss3: Hierarchical Triplet loss
            levels_sim = []
            embeds_0 = aigen_h_embedding[:, 0]
            for level in range(1, aigen_sent_num):
                embeds_1 = aigen_h_embedding[:, level]
                levels_sim.append(self.similarity(embeds_0, embeds_1))
            levels_sim = torch.stack(levels_sim, dim=-1)

            loss3_fn = F.relu # triplet
            loss3 = torch.tensor(0., device=self.device)
            if aigen_sent_num - 2:
                start = 0
                for idx in range(start, aigen_sent_num - 2):
                    loss3 += loss3_fn(levels_sim[:, idx + 1] - levels_sim[:, idx] + 
                                    margins[0][idx if idx < len(margins[0]) else len(margins[0]) - 1]).mean()
                        #    + loss3_fn(levels_sim[:, idx] - levels_sim[:, idx + 1] - 
                        #               margins[1][idx if idx < len(margins[1]) else len(margins[1]) - 1]).mean() # triplet
                loss3 /= (aigen_sent_num - 2) - start
        else:
            loss3 = None

        if use_loss4:
            loss4 = torch.tensor(0., device=self.device)
            for k in self.named_parameters_cache:
                if self.named_parameters_cache[k].device == self.device:
                    break
                self.named_parameters_cache[k] = self.named_parameters_cache[k].to(self.device)
            for k, v in self.named_parameters():
                if k in self.named_parameters_cache:
                    loss4 += torch.pow(v - self.named_parameters_cache[k], 2).sum()
            loss4 /= 2
        else:
            loss4 = None    

        loss = torch.tensor(0.).to(self.device)
        levels_similarity = None
        levels_outputs = outputs
        if loss4 is not None:
            self.loss_logs['loss4'].append(loss4.item())
            loss += self.config.gamma * loss4
        if loss2 is not None:
            beta2 = (aigen_batch_size) / (aigen_batch_size + other_batch_size)
            self.loss_logs['loss2'].append(loss2.item())
            loss += beta2 * loss2
        else:
            beta2 = 0.
        if loss3 is not None:
            self.loss_logs['loss3'].append(loss3.item())
            loss += self.config.beta * loss3
            levels_similarity = levels_sim

        if loss1 is not None:
            self.loss_logs['loss1'].append(loss1.item())
            loss += (1 - beta2) * loss1
            if levels_similarity is None:
                levels_similarity = similarity
        self.loss_logs['loss_all'].append(loss.item())
            
        if len(self.loss_logs['loss_all']) % 125 == 1:
            self.display_loss('avg')
        
        if not return_dict:
            output = (levels_similarity,) + \
                (levels_outputs.get("hidden_states", None), levels_outputs.get("attentions", None))
            return ((loss,) + output) if loss is not None else output
        return SequenceClassifierOutput(
            loss=loss,
            logits=levels_similarity,
            hidden_states=levels_outputs.get("hidden_states", None),
            attentions=levels_outputs.get("attentions", None),
        )


    def _sentemb_forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        loss_pair=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None
    ) -> BaseModelOutputWithPoolingAndCrossAttentions:
        return_dict: bool = return_dict if return_dict is not None else self.config.use_return_dict

        if isinstance(input_ids, list):
            input_ids = torch.cat([input_ids[0].reshape(-1, input_ids[0].shape[-1]), input_ids[1][:, 0]], dim=0)
            attention_mask = torch.cat([attention_mask[0].reshape(-1, attention_mask[0].shape[-1]), attention_mask[1][:, 0]], dim=0)
            token_type_ids = torch.cat([token_type_ids[0].reshape(-1, token_type_ids[0].shape[-1]), token_type_ids[1][:, 0]], dim=0)
        
        paired_data: bool = len(input_ids.shape) == 3

        if paired_data:
            batch_size = input_ids.size(0)
            # Number of sentences in one instance
            # 2: pair instance; 3: pair instance with a hard negative
            num_sent = input_ids.size(1)

            # Flatten input for encoding
            input_ids = input_ids.view((-1, input_ids.size(-1))) # (bs * num_sent, len)
            attention_mask = attention_mask.view((-1, attention_mask.size(-1))) # (bs * num_sent, len)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1))) # (bs * num_sent, len)
        
        # Get outputs
        with torch.no_grad():
            outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=True, # if cls.config.pooler_type in ['avg_top2', 'avg_first_last'] else False,
                return_dict=True,
            )
        
        # Get embeddings
        embedding: torch.Tensor = self._get_embedding(attention_mask, outputs, with_mlp=False) # FIXME, with_mlp=True
        if paired_data:
            embedding = embedding.view(batch_size, num_sent, embedding.shape[-1])

        if not return_dict:
            return (outputs.last_hidden_state, embedding, outputs.hidden_states)
        
        return BaseModelOutputWithPoolingAndCrossAttentions(
            pooler_output=embedding,
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states
        )

    def forward(self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            loss_pair=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            sent_emb=False
    ) -> Union[Tuple, Dict[str, Any]]:
        if 'mask' in self.config.pooler_type:
            input_ids, token_type_ids, attention_mask = self.add_prompt(input_ids, token_type_ids, attention_mask)

        forward_fn = self._sentemb_forward if sent_emb else self._ng_forward

        return forward_fn(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                loss_pair=loss_pair,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
        )

    def custom_param_init(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            logger.info(f'{self.config.num_layers} layers of Bert/Roberta are used for trainning.')
        unfreeze_layers = ['pooler'] + [f'layer.{11 - i}' for i in range(self.config.num_layers)]
        for name, param in self.bert.named_parameters():
            param.requires_grad = False
            for ele in unfreeze_layers:
                if ele in name:
                    param.requires_grad = True
                    break
        
        # nn.init.uniform_(self.mlp.weight, 0, 6e-3)
        # nn.init.uniform_(self.mlp.bias, 0, 6e-3)
        nn.init.normal_(self.mlp.weight, 0, 6e-3)
        nn.init.normal_(self.mlp.bias, 0, 6e-3)

        self.named_parameters_cache = {k: v.detach().clone() for k, v in self.named_parameters() 
                                       if 'bert' in k and v.requires_grad == True}
    
    def floating_point_ops(
        self, input_dict: Dict[str, Union[torch.Tensor, Any]], exclude_embeddings: bool = True
    ) -> int:
        def _fpo(
            input_dict: Dict[str, Union[torch.Tensor, Any]], exclude_embeddings: bool = True
        ) -> int:
            return 6 * self.estimate_tokens(input_dict) * self.num_parameters(exclude_embeddings=exclude_embeddings)
        
        total_fpo = 0.
        list_len = 0
        for v in input_dict.values():
            if isinstance(v, list):
                list_len = len(v)
                break
        
        if list_len:
            for i in range(list_len):
                tmp_inp = {k: (v[i] if isinstance(v, list) else v) for k, v in input_dict.items()}
                total_fpo += _fpo(tmp_inp, exclude_embeddings)
        else:
            total_fpo = _fpo(input_dict, exclude_embeddings)
        
        return total_fpo
    
class NGRoberta(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = ['pooler', 'cls']
    config_class = NGRobertaConfig

    def __init__(self, config, *model_args, **model_kwargs):
        super().__init__(config)
        # self.model_args = model_kwargs["model_args"]
        self.roberta = RobertaModel(config, add_pooling_layer=False)

        self.prompt = [
            {
                'input_ids': [torch.tensor([713, 3645, 4832, 22]), torch.tensor([113, 839, 50264])],
                'attention_mask': [torch.tensor([1, 1, 1, 1]), torch.tensor([1, 1, 1])]
            },
            {
                'input_ids': [torch.tensor([713, 3645, 9, 22]), torch.tensor([113, 839, 50264])],
                'attention_mask': [torch.tensor([1, 1, 1, 1]), torch.tensor([1, 1, 1])]
            }
        ]

        self.mlp = nn.Linear(config.hidden_size, config.feature_size)
        self.similarity = lambda x,y: F.cosine_similarity(x, y, dim=-1)

        self.named_parameters_cache = {k: v.detach().clone() for k, v in self.named_parameters() 
                                       if 'roberta' in k and v.requires_grad == True}

        self.loss_logs = {'loss_all': [], 'loss1': [], 'loss2': [], 'loss3': [], 'loss4': []}

    def add_prompt(self, input_ids, token_type_ids, attention_mask):
        def _add_prompt(prompt, input_ids, token_type_ids, attention_mask):
            sent_len = attention_mask.sum(dim=-1)
            n_input_ids = []
            # if token_type_ids is not None:
            #     n_token_type_ids = []
            n_attention_mask = []
            for s_idx in range(input_ids.shape[0]):
                n_input_ids.append(torch.cat([input_ids[s_idx][:1], prompt['input_ids'][0].to(input_ids[s_idx][:1]), 
                    input_ids[s_idx][1:sent_len[s_idx] - 1], prompt['input_ids'][1].to(input_ids[s_idx][:1]), 
                    input_ids[s_idx][sent_len[s_idx] - 1: ]], dim=0))
                # if token_type_ids is not None:
                #     n_token_type_ids.append(torch.cat([token_type_ids[s_idx][:1], prompt['token_type_ids'][0].to(token_type_ids[s_idx][:1]), 
                #         token_type_ids[s_idx][1:sent_len[s_idx] - 1], prompt['token_type_ids'][1].to(token_type_ids[s_idx][:1]), 
                #         token_type_ids[s_idx][sent_len[s_idx] - 1: ]], dim=0))
                n_attention_mask.append(torch.cat([attention_mask[s_idx][:1], prompt['attention_mask'][0].to(attention_mask[s_idx][:1]),
                    attention_mask[s_idx][1:sent_len[s_idx] - 1], prompt['attention_mask'][1].to(attention_mask[s_idx][:1]), 
                    attention_mask[s_idx][sent_len[s_idx] - 1: ]], dim=0))
            n_input_ids = torch.stack(n_input_ids, dim=0)
            # if token_type_ids is not None:
            #     n_token_type_ids = torch.stack(n_token_type_ids, dim=0)
            # else:
            #     n_token_type_ids = None
            n_attention_mask = torch.stack(n_attention_mask, dim=0)
            return n_input_ids, None, n_attention_mask
        
        if isinstance(input_ids, list):

            if input_ids[0].shape[0] > 0:
                n_input_ids0 = []
                # if token_type_ids[0] is not None:
                #     n_token_type_ids0 = []
                n_attention_mask0 = []
                for idx1 in range(input_ids[0].shape[1]):
                    r_input_ids, r_token_type_ids, r_attention_mask = _add_prompt(
                        self.prompt[0], input_ids[0][:, idx1], None,
                        attention_mask[0][:, idx1]
                    )
                    n_input_ids0.append(r_input_ids)
                    # if token_type_ids[0] is not None:
                    #     n_token_type_ids0.append(r_token_type_ids)
                    n_attention_mask0.append(r_attention_mask)
                input_ids[0] = torch.stack(n_input_ids0, dim=1)
                # if token_type_ids[0] is not None:
                #     token_type_ids[0] = torch.stack(n_token_type_ids0, dim=1)
                # else:
                #     token_type_ids[0] = None
                attention_mask[0] = torch.stack(n_attention_mask0, dim=1)

            if input_ids[1].shape[0] > 0:
                n_input_ids1 = []
                # if token_type_ids[1] is not None:
                #     n_token_type_ids1 = []
                n_attention_mask1 = []
                for idx1 in range(input_ids[1].shape[1]):
                    if input_ids[1].shape[1] == 2:
                        r_input_ids, r_token_type_ids, r_attention_mask = _add_prompt(
                            self.prompt[idx1], input_ids[1][:, idx1], None,
                            attention_mask[1][:, idx1]
                        )
                    else:
                        r_input_ids, r_token_type_ids, r_attention_mask = _add_prompt(
                            self.prompt[0], input_ids[1][:, idx1], None,
                            attention_mask[1][:, idx1]
                        )
                    n_input_ids1.append(r_input_ids)
                    # if token_type_ids[1] is not None:
                    #     n_token_type_ids1.append(r_token_type_ids)
                    n_attention_mask1.append(r_attention_mask)
                input_ids[1] = torch.stack(n_input_ids1, dim=1)
                # if token_type_ids[1] is not None:
                #     token_type_ids[1] = torch.stack(n_token_type_ids1, dim=1)
                # else:
                #     token_type_ids[1] = None
                attention_mask[1] = torch.stack(n_attention_mask1, dim=1)
            
            if input_ids[0].shape[-1] != input_ids[1].shape[-1]:
                if input_ids[0].shape[0] == 0:
                    input_ids[0] = input_ids[0].reshape(*(input_ids[0].shape[:-1] + input_ids[1].shape[-1:]))
                    # if token_type_ids[0] is not None:
                    #     token_type_ids[0] = token_type_ids[0].reshape(*(token_type_ids[0].shape[:-1] + token_type_ids[1].shape[-1:]))
                    attention_mask[0] = attention_mask[0].reshape(*(attention_mask[0].shape[:-1] + attention_mask[1].shape[-1:]))
                elif input_ids[1].shape[0] == 0:
                    input_ids[1] = input_ids[1].reshape(*(input_ids[1].shape[:-1] + input_ids[0].shape[-1:]))
                    # if token_type_ids[1] is not None:
                    #     token_type_ids[1] = token_type_ids[1].reshape(*(token_type_ids[1].shape[:-1] + token_type_ids[0].shape[-1:]))
                    attention_mask[1] = attention_mask[1].reshape(*(attention_mask[1].shape[:-1] + attention_mask[0].shape[-1:]))
                else:
                    raise Exception

        else:

            if len(input_ids.shape) == 3:
                n_input_ids = []
                # if token_type_ids is not None:
                #     n_token_type_ids = []
                n_attention_mask = []

                for idx1 in range(input_ids.shape[1]):
                    r_input_ids, r_token_type_ids, r_attention_mask = _add_prompt(
                        self.prompt[idx1 % 2], input_ids[:, idx1], None, 
                        attention_mask[:, idx1]
                    )
                    n_input_ids.append(r_input_ids)
                    # if token_type_ids is not None:
                    #     n_token_type_ids.append(r_token_type_ids)
                    n_attention_mask.append(r_attention_mask)
                input_ids = torch.stack(n_input_ids, dim=1)
                # if token_type_ids is not None:
                #     token_type_ids = torch.stack(n_token_type_ids, dim=1)
                attention_mask = torch.stack(n_attention_mask, dim=1)

            else:
                input_ids, token_type_ids, attention_mask = _add_prompt(
                    self.prompt[0], input_ids, None, attention_mask
                )

        return input_ids, token_type_ids, attention_mask

    def display_loss(self, mode: Union[int, str]):
        if dist.is_initialized() and dist.get_rank() != 0:
            return
        if mode == 'avg':
            logger.info("Average loss:\n{}".format(
                {k: sum(v) / len(v) for k, v in self.loss_logs.items() if v}
            ))
        elif isinstance(mode, int):
            logger.info("Loss every {} inputs:\n{}".format(
                mode, {
                    k: [(i_loss, sum(v[: i_index + 1]) / (i_index + 1)) for i_index, i_loss in enumerate(v) if (i_index % mode) == 0] 
                    for k, v in self.loss_logs.items() if v
                }
            ))
        else:
            logger.error(f"Can not display loss in mode {mode}")

    def _get_embedding(
        self, attention_mask: torch.Tensor, outputs: Dict[str, torch.Tensor],
        with_mlp: bool = True
    ) -> torch.Tensor:
        last_hidden: torch.Tensor = outputs.last_hidden_state # (bs, seq_len, hidden_len)
        hidden_states: torch.Tensor = outputs.hidden_states  # Tuple of (bs, seq_len, hidden_len)

        if 'cls' in self.config.pooler_type:
            cls_embedding = last_hidden[:, 0]
            if with_mlp:
                return self.mlp(cls_embedding) #(bs, hidden_len)
            else:
                return cls_embedding
        elif 'avg' in self.config.pooler_type:
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / 
                     attention_mask.sum(-1).unsqueeze(-1)) #(bs, hidden_len)
        elif 'mask' in self.config.pooler_type:
            mask_embedding = []
            sent_len = attention_mask.sum(dim=-1) 
            for idx in range(last_hidden.shape[0]):
                mask_embedding.append(last_hidden[idx][sent_len[idx] - 2])
            mask_embedding = torch.stack(mask_embedding, dim=0)
            if with_mlp:
                return self.mlp(mask_embedding) #(bs, hidden_len)
            else:
                return mask_embedding
        else:
            raise NotImplementedError

    def _ng_forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        loss_pair=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None
    ) -> SequenceClassifierOutput:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        ori_input_ids = input_ids


        aigen_batch_size = input_ids[0].size(0)
        aigen_sent_num = input_ids[0].size(1)
        other_batch_size = input_ids[1].size(0)
        other_sent_num = input_ids[1].size(1)

        
        inp_input_ids = torch.cat([input_ids[0].reshape(-1, input_ids[0].shape[-1]), 
                                       input_ids[1].reshape(-1, input_ids[1].shape[-1])], dim=0) # shape of [abs * aigen_sent_num + obs * 2, seq_len]
        inp_attention_mask = torch.cat([attention_mask[0].reshape(-1, attention_mask[0].shape[-1]),
                                        attention_mask[1].reshape(-1, attention_mask[1].shape[-1])], dim=0)
        if token_type_ids is not None and token_type_ids[1] is not None:
            inp_token_type_ids = torch.cat([token_type_ids[0].reshape(-1, token_type_ids[0].shape[-1]),
                                            token_type_ids[1].reshape(-1, token_type_ids[1].shape[-1])], dim=0)
        else:
            inp_token_type_ids = None

        # Get raw embeddings
        outputs = self.roberta(
            inp_input_ids,
            attention_mask=inp_attention_mask,
            token_type_ids=inp_token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True, # if cls.config.pooler_type in ['avg_top2', 'avg_first_last'] else False,
            return_dict=True,
        )

        embedding = self._get_embedding(inp_attention_mask, outputs, with_mlp=True) # (abs * aigen_sent_num + obs * 2, hidden_size)

        if dist.is_initialized() and self.training:
            raise NotImplementedError # FIXME unavailable when abs and obs are not fixed between batches.

        aigen_h_embedding = embedding[:aigen_batch_size * aigen_sent_num].reshape(aigen_batch_size, aigen_sent_num, embedding.shape[-1])
        other_h_embedding = embedding[aigen_batch_size * aigen_sent_num:].reshape(other_batch_size, other_sent_num, embedding.shape[-1])
        
        temp, margins = self.config.temp, self.config.margins
        use_loss4 = self.config.use_loss4
        use_loss3 = self.config.use_loss3
        use_native_loss = self.config.use_native_loss

        if use_native_loss:
            
            loss1_fn = nn.CrossEntropyLoss()

            aigen_pos_idx = 2 if aigen_sent_num > 3 else 1

            # embeds_0 = torch.cat([other_h_embedding[:, :1], aigen_h_embedding[:, :1]], dim=0)
            # embeds_1 = torch.cat([other_h_embedding[None, :, 1], aigen_h_embedding[None, :, aigen_pos_idx]], dim=1)                
            
            embeds_0 = torch.cat(
                ([other_h_embedding[:, :1]] if other_sent_num else []) + 
                ([aigen_h_embedding[:, :1]] if aigen_sent_num else []),
                dim=0
            )
            embeds_1 = torch.cat(
                ([other_h_embedding[None, :, 1]] if other_sent_num else []) +
                ([aigen_h_embedding[None, :, aigen_pos_idx]] if aigen_sent_num else []) +
                ([other_h_embedding[None, :, -1]] if other_sent_num > 2 else []) +
                ([aigen_h_embedding[None, :, -1]] if aigen_sent_num > 2 and not use_loss3 else []),
                dim=1
            )


            similarity = self.similarity(embeds_0, embeds_1)
            labels = torch.arange(similarity.shape[0]).to(dtype=torch.long, device=self.device)
            loss1 = loss1_fn(similarity / temp, labels)

            loss2 = None

        else:

            if other_batch_size > 0:
                # loss1: CrossEntropy or InforNCE
                loss1_fn = nn.CrossEntropyLoss()
                embeds_0 = other_h_embedding[:, :1] # (obs, 1, hidden_size)
                embeds_1 = torch.cat([other_h_embedding[:, 1], 
                                      aigen_h_embedding[:, 1]] + 
                                     ([] if other_sent_num == 2 else [other_h_embedding[:, -1]])
                                    , dim=0).unsqueeze(0) # (1, obs + abs * asn, hidden_size)
                # embeds_1 = other_h_embedding[None, :, 1] # alternative
                similarity = self.similarity(embeds_0, embeds_1) # (obs, obs + abs * asn)
                labels = torch.arange(similarity.shape[0]).to(dtype=torch.long, device=self.device)
                loss1 = loss1_fn(similarity / temp, labels)
            else:
                loss1 = None
            
            
            if aigen_batch_size > 0:
                # loss2: CrossEntropy or InforNCE
                loss2_fn = nn.CrossEntropyLoss()
                loss2 = torch.tensor(0., device=self.device)
                aigen_pos_sent_num = 2 # FIXME
                embeds_0 = aigen_h_embedding[:, :1] # (abs, 1, hidden_size)
                
                for aigen_pos_idx in range(1, aigen_pos_sent_num + 1):
                    if aigen_pos_idx == aigen_pos_sent_num:
                        embeds_1 = torch.cat([aigen_h_embedding[:, aigen_pos_idx], aigen_h_embedding[:, -1], other_h_embedding[:, 1]], dim=0).unsqueeze(0)
                    else:
                        embeds_1 = torch.cat([aigen_h_embedding[:, aigen_pos_idx], other_h_embedding[:, 1]], dim=0).unsqueeze(0) # (1, abs + obs, hidden_size)
                    # w/o hn
                    # embeds_1 = torch.cat([aigen_h_embedding[:, aigen_pos_idx], other_h_embedding[:, 1]], dim=0).unsqueeze(0) # (1, abs + obs, hidden_size)
                    # w hn: -1
                    # embeds_1 = torch.cat([aigen_h_embedding[:, aigen_pos_idx], aigen_h_embedding[:, -1], other_h_embedding[:, 1]], dim=0).unsqueeze(0) # (1, 2 * abs + obs, hidden_size)
                    
                    similarity = self.similarity(embeds_0, embeds_1) # (abs, obs + abs)
                    labels = torch.arange(similarity.shape[0]).to(dtype=torch.long, device=self.device)
                    loss2 += loss2_fn(similarity / temp, labels)
                loss2 /= aigen_pos_sent_num
    
            else:
                loss2 = None

        if use_loss3 and aigen_batch_size > 0:
            # loss3: Hierarchical Triplet loss
            levels_sim = []
            embeds_0 = aigen_h_embedding[:, 0]
            for level in range(1, aigen_sent_num):
                embeds_1 = aigen_h_embedding[:, level]
                levels_sim.append(self.similarity(embeds_0, embeds_1))
            levels_sim = torch.stack(levels_sim, dim=-1)

            loss3_fn = F.relu # triplet
            loss3 = torch.tensor(0., device=self.device)
            if aigen_sent_num - 2:
                start = 0
                for idx in range(start, aigen_sent_num - 2):
                    loss3 += loss3_fn(levels_sim[:, idx + 1] - levels_sim[:, idx] + 
                                    margins[0][idx if idx < len(margins[0]) else len(margins[0]) - 1]).mean()
                        #    + loss3_fn(levels_sim[:, idx] - levels_sim[:, idx + 1] - 
                        #               margins[1][idx if idx < len(margins[1]) else len(margins[1]) - 1]).mean() # triplet
                loss3 /= (aigen_sent_num - 2) - start
        else:
            loss3 = None


        if use_loss4:
            loss4 = torch.tensor(0., device=self.device)
            for k in self.named_parameters_cache:
                if self.named_parameters_cache[k].device == self.device:
                    break
                self.named_parameters_cache[k] = self.named_parameters_cache[k].to(self.device)
            for k, v in self.named_parameters():
                if k in self.named_parameters_cache:
                    loss4 += torch.pow(v - self.named_parameters_cache[k], 2).sum()
            loss4 /= 2
        else:
            loss4 = None    

        loss = torch.tensor(0.).to(self.device)
        levels_similarity = None
        levels_outputs = outputs
        if loss4 is not None:
            self.loss_logs['loss4'].append(loss4.item())
            loss += self.config.gamma * loss4
        if loss2 is not None:
            beta2 = (aigen_batch_size) / (aigen_batch_size + other_batch_size)
            self.loss_logs['loss2'].append(loss2.item())
            loss += beta2 * loss2
        else:
            beta2 = 0.
        if loss3 is not None:
            self.loss_logs['loss3'].append(loss3.item())
            loss += self.config.beta * loss3
            levels_similarity = levels_sim

        if loss1 is not None:
            self.loss_logs['loss1'].append(loss1.item())
            loss += (1 - beta2) * loss1
            if levels_similarity is None:
                levels_similarity = similarity
        self.loss_logs['loss_all'].append(loss.item())
            
        if len(self.loss_logs['loss_all']) % 125 == 1:
            self.display_loss('avg')
        
        if not return_dict:
            output = (levels_similarity,) + \
                (levels_outputs.get("hidden_states", None), levels_outputs.get("attentions", None))
            return ((loss,) + output) if loss is not None else output
        return SequenceClassifierOutput(
            loss=loss,
            logits=levels_similarity,
            hidden_states=levels_outputs.get("hidden_states", None),
            attentions=levels_outputs.get("attentions", None),
        )


    def _sentemb_forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        loss_pair=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None
    ) -> BaseModelOutputWithPoolingAndCrossAttentions:
        return_dict: bool = return_dict if return_dict is not None else self.config.use_return_dict

        if isinstance(input_ids, list):
            input_ids = torch.cat([input_ids[0].reshape(-1, input_ids[0].shape[-1]), input_ids[1][:, 0]], dim=0)
            attention_mask = torch.cat([attention_mask[0].reshape(-1, attention_mask[0].shape[-1]), attention_mask[1][:, 0]], dim=0)
            # token_type_ids = torch.cat([token_type_ids[0].reshape(-1, token_type_ids[0].shape[-1]), token_type_ids[1][:, 0]], dim=0)
        
        paired_data: bool = len(input_ids.shape) == 3

        if paired_data:
            batch_size = input_ids.size(0)
            # Number of sentences in one instance
            # 2: pair instance; 3: pair instance with a hard negative
            num_sent = input_ids.size(1)

            # Flatten input for encoding
            input_ids = input_ids.view((-1, input_ids.size(-1))) # (bs * num_sent, len)
            attention_mask = attention_mask.view((-1, attention_mask.size(-1))) # (bs * num_sent, len)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1))) # (bs * num_sent, len)
        
        # Get outputs
        with torch.no_grad():
            outputs = self.roberta(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=True, # if cls.config.pooler_type in ['avg_top2', 'avg_first_last'] else False,
                return_dict=True,
            )
        
        # Get embeddings
        embedding: torch.Tensor = self._get_embedding(attention_mask, outputs, with_mlp=False) # FIXME, with_mlp=True
        if paired_data:
            embedding = embedding.view(batch_size, num_sent, embedding.shape[-1])

        if not return_dict:
            return (outputs.last_hidden_state, embedding, outputs.hidden_states)
        
        return BaseModelOutputWithPoolingAndCrossAttentions(
            pooler_output=embedding,
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states
        )

    def forward(self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            loss_pair=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            sent_emb=False
    ) -> Union[Tuple, Dict[str, Any]]:
        if 'mask' in self.config.pooler_type:
            input_ids, token_type_ids, attention_mask = self.add_prompt(input_ids, token_type_ids, attention_mask)

        forward_fn = self._sentemb_forward if sent_emb else self._ng_forward

        return forward_fn(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                loss_pair=loss_pair,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
        )

    def custom_param_init(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            logger.info(f'{self.config.num_layers} layers of Bert/Roberta are used for trainning.')
        unfreeze_layers = ['pooler'] + [f'layer.{11 - i}' for i in range(self.config.num_layers)]
        for name, param in self.roberta.named_parameters():
            param.requires_grad = False
            for ele in unfreeze_layers:
                if ele in name:
                    param.requires_grad = True
                    break
        
        # nn.init.uniform_(self.mlp.weight, 0, 6e-3)
        # nn.init.uniform_(self.mlp.bias, 0, 6e-3)
        nn.init.normal_(self.mlp.weight, 0, 6e-3)
        nn.init.normal_(self.mlp.bias, 0, 6e-3)

        self.named_parameters_cache = {k: v.detach().clone() for k, v in self.named_parameters() 
                                       if 'roberta' in k and v.requires_grad == True}
    
    def floating_point_ops(
        self, input_dict: Dict[str, Union[torch.Tensor, Any]], exclude_embeddings: bool = True
    ) -> int:
        def _fpo(
            input_dict: Dict[str, Union[torch.Tensor, Any]], exclude_embeddings: bool = True
        ) -> int:
            return 6 * self.estimate_tokens(input_dict) * self.num_parameters(exclude_embeddings=exclude_embeddings)
        
        total_fpo = 0.
        list_len = 0
        for v in input_dict.values():
            if isinstance(v, list):
                list_len = len(v)
                break
        
        if list_len:
            for i in range(list_len):
                tmp_inp = {k: (v[i] if isinstance(v, list) else v) for k, v in input_dict.items()}
                total_fpo += _fpo(tmp_inp, exclude_embeddings)
        else:
            total_fpo = _fpo(input_dict, exclude_embeddings)
        
        return total_fpo