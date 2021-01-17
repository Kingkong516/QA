import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import (AlbertPreTrainedModel, AlbertTokenizer, AlbertModel)
from dataclasses import dataclass
from transformers.file_utils import ModelOutput
from typing import Optional, Tuple

from utils import move_to_device

class QAModel(nn.Module):
    
    def __init__(self, cuda=False):
        super(QAModel, self).__init__()
        
        self.cuda = cuda
        self.device = 'cuda' if cuda else 'cpu'
        self.tokenizer =  AlbertTokenizer.from_pretrained('albert-base-v2')
        self.model = QAAlbert.from_pretrained('albert-base-v2', return_dict=True)
        self.max_len = self.model.max_len
        
        if self.cuda:
            self.model.to('cuda')

    def forward(self, data):
        data = {k:v for k,v in data.items() if k!='q_idx'} # question index is useless for model
        data = move_to_device(data, self.device)        
        output = self.model(**data, return_dict=True)
        return output
    
    def predict(self, data):
        output = self.forward(data)
        unk = torch.argmax(output.unk_logits,axis=-1)
        start_positions = torch.argmax(output.start_logits,axis=-1)
        end_positions = torch.argmax(output.end_logits,axis=-1)
        start_positions[unk==1] = -1
        end_positions[unk==1] = -1
        if self.cuda:
            del output
            torch.cuda.empty_cache()
        return (start_positions, end_positions)
    
class QAAlbert(AlbertPreTrainedModel):
    """Customized from AlbertForQuestionAnswering
    """
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.albert = AlbertModel(config)
        self.unk_outputs = nn.Linear(config.hidden_size, 2)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.max_len = self.albert.config.max_position_embeddings
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
            Index -1 is ignored (treated as indicator for no answer).
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
            Index -1 is ignored (treated as indicator for no answer).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.albert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        sequence_output = outputs[0] # last hidden state
        pooled_output = outputs[1] # first along the sequence
        total_loss = None
        
        # loss on the unk
        unk_logits = self.unk_outputs(pooled_output)
        if start_positions is not None and end_positions is not None:
            labels = (start_positions == -1).long()
            unk_loss_fct = CrossEntropyLoss()
            unk_loss = unk_loss_fct(unk_logits, labels)
            total_loss = 2*unk_loss # more weight on UNK?
        
        # loss on the start and end positions
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
               
        # loss on the position
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(-1, ignored_index)
            end_positions.clamp_(-1, ignored_index)
            start_positions[start_positions==ignored_index] = -1
            end_positions[end_positions==ignored_index] = -1

            loss_fct = CrossEntropyLoss(ignore_index=-1)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss += ((start_loss + end_loss) / 2)
                
        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QAModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            unk_logits=unk_logits,
            hidden_states=outputs.last_hidden_state,
            attentions=outputs.attentions,
        )
        
@dataclass
class QAModelOutput(ModelOutput):
    """
    Base class for outputs of question answering models.

    Args:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
        unk_logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, 2,)`):
            whether the answer is unknown (before SoftMax).
        start_logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length,)`):
            Span-start scores (before SoftMax).
        end_logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length,)`):
            Span-end scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    unk_logits: torch.FloatTensor = None
    start_logits: torch.FloatTensor = None
    end_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None        
    
            
        
    