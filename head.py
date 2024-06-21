from typing import Tuple, Any

import torch
import torch.nn.functional as F
import torch.nn as nn
import random

from typing import Optional, Union


class SelfRewardHead(nn.Module):
    def __init__(self, in_features: int, out_features: int = 1, bias: bool = False):
        super(SelfRewardHead, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # You can use the LinearMoE implementation from my repository: https://github.com/mkurman/linearmoe_pytorch
        self.reward_head = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, embedding_module: nn.Module, input_ids: torch.FloatTensor, output: Optional[torch.FloatTensor] = None) -> tuple[
        torch.FloatTensor, torch.FloatTensor]:
        with torch.no_grad():
            reward_outputs = []
            reward_labels = []
            for i in range(input_ids.shape[0]):
                if output is None:
                    output_to_choose = 1
                else:
                    output_to_choose = random.randint(0, 1)

                if output_to_choose == 0:
                    reward_input = output.argmax(dim=-1)
                else:
                    reward_input = input_ids
                    output_to_choose = 1.

                reward_input = embedding_module(torch.stack([reward_input[i]]))[0]

                if output_to_choose == 0:
                    similarity_score = nn.CosineSimilarity(dim=1, eps=1e-6)
                    reward_input_gold = embedding_module(torch.stack([input_ids[i]]))[0]
                    output_to_choose = similarity_score(reward_input, reward_input_gold)[0].detach().cpu().numpy()

                s, d = reward_input.size()
                mask = torch.ones((s, s), dtype=torch.bool, device=reward_input.device).triu(1)
                ones = torch.ones((s, s), dtype=reward_input.dtype, device=reward_input.device)

                summator = ones.masked_fill(mask, 0.)

                reward_input = summator @ reward_input

                reward_outputs.append(reward_input[..., -1:, :])
                reward_labels.append(torch.tensor(float(output_to_choose)))

            reward_outputs = torch.stack(reward_outputs)
            reward_labels = torch.stack(reward_labels)

        reward_output = self.reward_head(reward_outputs)

        return reward_output, reward_labels


class AnyModelForCausalLM(AnyModel):
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config: AnyConfig):
        super().__init__(config)

        # ... previous code

        # This is the most important layer. You can use a simple Linear layer instead.
        self.rewarding_lm_head = SelfRewardHead(config.hidden_size, 1, bias=False)

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
                ) -> Union[Tuple, Any]:
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

        loss = None

        if labels is not None:
            # This is the most important part of the training phase. It calculates the loss and the self-reward loss
            # Provide embeddings layer and input_ids as the "ground truth", and the model outputs as "logits" to the rewarding_lm_head
            reward_output, reward_labels = self.rewarding_lm_head(self.model.embeddings, input_ids, logits)

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

        # ... rest of the code

    # Self-rewarding method that allows you to rate any tokenized text
    def calc_reward(self, input_ids):
        with torch.no_grad():
            return F.sigmoid(self.rewarding_lm_head(input_ids)[0])
