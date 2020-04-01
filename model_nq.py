from transformers import BertTokenizer, AdamW, BertModel, BertPreTrainedModel
import torch.nn as nn

class BertForQuestionAnswering(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForQuestionAnswering, self).__init__(config)
        self.bert = BertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)  # start/end
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, labels=None):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids, 
                            head_mask=head_mask)

        sequence_output = outputs[0]
        pooled_output = outputs[1]

        # predict start & end position
        sequence_output = self.dropout(sequence_output)
        qa_logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = qa_logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
    
        # classification
        pooled_output = self.dropout(pooled_output)
        classifier_logits = self.classifier(pooled_output)
        
        if labels is not None:
            start_labels, end_labels, class_labels = labels
            start_loss = nn.CrossEntropyLoss(ignore_index=-1)(start_logits, start_labels)
            end_loss = nn.CrossEntropyLoss(ignore_index=-1)(end_logits, end_labels)
            class_loss = nn.CrossEntropyLoss()(classifier_logits, class_labels)
            outputs = start_loss + end_loss + 2*class_loss
        else:
            outputs = (start_logits, end_logits, classifier_logits)

        return outputs