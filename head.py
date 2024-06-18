import torch
import torch.nn.functional as F

# --- This is just a fragment of the model, the most important part is reward_lm_head and its implementation in the forward() function --- #
# --- You can find the LinearMoE implementation in my repository: https://github.com/mkurman/linearmoe_pytorch --- #

class KurmanForCausalLM(KurmanPreTrainedModel):
  _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

  def __init__(self, config: KurmanConfig):
      super().__init__(config)

      self.config = config

      self.model = KurmanModel(config)
      self.lm_head = LinearMoE(config.hidden_size, config.vocab_size,  bias=False, num_experts=config.num_experts, top_k=config.top_k_experts, r=config.experts_r)

      # This is the most important layer. You can use a simple Linear layer instead.
      self.rewarding_lm_head = LinearMoE(config.hidden_size, 1, bias=False, num_experts=config.num_experts, top_k=config.top_k_experts, r=config.experts_r)

      self.reset_parameters()

  def reset_parameters(self):
      # ...

  def forward(self,
              input_ids: torch.LongTensor = None,
              attention_mask=None,
              labels: Optional[torch.LongTensor] = None,
              past_key_values=None,
              position_ids=None,
              token_type_ids=None,
              output_attentions=None,
              output_hidden_states=None,
              use_cache: Optional[bool] = None,
              return_dict: Optional[bool] = None,
              **kwargs,
              ) -> Union[Tuple, RewardCausalLMOutputWithPast]:
      return_dict = return_dict if return_dict is not None else self.config.use_return_dict

      outputs = self.model(
          input_ids=input_ids,
          attention_mask=attention_mask,
          use_cache=use_cache,
          return_dict=return_dict,
          past_key_values=past_key_values,
          **kwargs, )

      hidden_state = outputs[0]

      logits = self.lm_head(hidden_state)

      if len(outputs) > 1:
          past_key_values = outputs[1]
  
      loss = None
                
      reward_loss = None
      reward_labels = None
      reward_outputs = None
                
      if labels is not None:
        # This is the most important part of the training phase.
        # -> Preparing inputs for self reward
        with torch.no_grad():
            reward_outputs = []
            reward_labels = []
            for i in range(input_ids.shape[0]):
                output_to_choose = random.randint(0, 1)
    
                if output_to_choose == 0:
                    reward_input = outputs[0].argmax(dim=-1)
                else:
                    reward_input = input_ids
  
                reward_input = self.model.embeddings(torch.stack([reward_input[i]]))[0]
  
                s, d = reward_input.size()
                mask = torch.ones((s, s), dtype=torch.bool, device=reward_input.device).triu(1)
                ones = torch.ones((s, s), dtype=reward_input.dtype, device=reward_input.device)
        
                summator = ones.masked_fill(mask, 0.)
        
                reward_input = summator @ reward_input
                
                reward_outputs.append(reward_input[..., -1:, :])
                reward_labels.append(torch.tensor(float(output_to_choose)))
  
            reward_outputs = torch.stack(reward_outputs) 
            reward_labels = torch.stack(reward_labels)
  
        reward_output = self.rewarding_lm_head(reward_outputs)

        # calc causal loss
        loss_fct = nn.CrossEntropyLoss(reduction='mean')
        
        loss_logits = logits[..., :-1, :].contiguous()
        loss_labels = labels[..., 1:].contiguous()
        
        loss_logits = loss_logits.view(-1, self.config.vocab_size)
        loss_labels = loss_labels.view(-1)
        
        loss_labels = loss_labels.to(loss_logits.device)
        loss = loss_fct(loss_logits, loss_labels)

        # calc self reward loss
        loss_fct = nn.BCEWithLogitsLoss(reduction='mean')
        
        loss_logits = reward_output[..., :].contiguous()
        loss_labels = reward_labels[..., :].contiguous()

        loss_logits = loss_logits.view(-1)
        loss_labels = loss_labels.view(-1)
        
        loss_labels = loss_labels.to(loss_logits.device)
        
        reward_loss = loss_fct(loss_logits, loss_labels)

        # sum losses
        loss = loss + reward_loss

      if not return_dict:
          output = (logits,) + outputs[1:]
          return loss, output, past_key_values

      return RewardCausalLMOutputWithPast(
          loss=loss,
          logits=logits,
          past_key_values=past_key_values,
          reward_logits=reward_output,
          reward_loss=reward_loss,
          reward_labels=reward_labels,
      )

  # ...

  # Self-rewarding method that allows you to rate any tokenized text
  def calc_reward(self, input_ids):
    with torch.no_grad():
        reward_input = self.model.embeddings(input_ids)
        
        b, s, d = reward_input.size()
        
        mask = torch.ones((s, s), dtype=torch.bool, device=reward_input.device).triu(1)
        ones = torch.ones((s, s), dtype=reward_input.dtype, device=reward_input.device)
  
        summator = ones.masked_fill(mask, 0.)
  
        reward_input = summator @ reward_input
            
        return F.sigmoid(self.rewarding_lm_head(reward_input[:, -1:]))

@dataclass
class RewardCausalLMOutputWithPast(CausalLMOutputWithPast):
  loss: Optional[torch.FloatTensor] = None
  logits: torch.FloatTensor = None
  past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
  hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
  attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
  reward_logits: Optional[torch.FloatTensor] = None
  reward_loss: Optional[torch.FloatTensor] = None
  reward_labels: Optional[torch.FloatTensor] = None
  
