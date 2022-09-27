#!/usr/bin/env python

import torch
from torch import nn

from transformers import BertForTokenClassification

from torchcrf import CRF #<------ replaced it with line below, as torchcrf was hard/impossible to install; eventually did `pip install pytorch-crf' to make this line work
#from TorchCRF import CRF #tried this as well, however it requires different arguments

class TokenBERT(nn.Module):
    '''
        Token BERT with (optional) Conditional Random Fileds (CRF)
    '''

    def __init__(self, num_labels, model_name, output_hidden_states=False,
            output_attentions=False, batch_first=True, use_crf=True):
        super(TokenBERT, self).__init__()
        self.num_labels = num_labels
        self.batch_first = batch_first
        self.use_crf = use_crf
        self.tokenbert = BertForTokenClassification.from_pretrained(
            model_name,
            num_labels=self.num_labels,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions
        )
        if self.use_crf:
            self.crf = CRF(self.num_labels, batch_first=self.batch_first)

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        outputs = self.tokenbert.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        sequence_output = outputs[0]
        #print("sequence_output:",sequence_output)
        sequence_output = self.tokenbert.dropout(sequence_output)  # (B, L, H)
        #print("sequence_output2:",sequence_output)
        logits = self.tokenbert.classifier(sequence_output)
        #print("logits:",logits)
        #print("shape logits",logits.shape)
        #print("shape labels",labels.shape)        
        #print("shape attention_mask",attention_mask.shape)

        if self.use_crf:
            if labels is not None: # training
                return -self.crf(logits, labels, attention_mask.byte())
            else: # inference
                return self.crf.decode(logits, attention_mask.byte())
        else:
            if labels is not None: # training
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    logits.view(-1, self.num_labels),
                    labels.view(-1)
                )
                return loss
            else: # inference
                return torch.argmax(logits, dim=2)
