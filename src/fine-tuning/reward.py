import json
import os
from typing import Optional, List, Iterable, Dict, Any, Tuple
import torch, spacy
import torch.nn.functional as F
from transformers import AutoTokenizer
import numpy as np
import logging
import re

from fgrlhf.reward import BasicReward
from fgrlhf.reward_utils import split_text_to_subsentences, split_text_to_sentences
#from fgrlhf.evaluators import get_rouge_scores

from my_longformer import LongformerForSequenceClassification, LongformerForTokenClassification, LongformerForTokenMultiClassification

logging.basicConfig(level=logging.ERROR)

class FactualityReward:
    def __init__(self,
                 tokenizer,
                 model_ckpt,
                 factuality_positive_reward = 1.0,
                 factuality_negative_reward = -1.0,
                 sep = "</s>",
                 ):
        
        # prepare policy tokenizer
        self.policy_tokenizer = tokenizer

        # prepare reward tokenizer
        self.reward_tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
        
        # prepare factual reward model
        self.f_reward_model = LongformerForTokenClassification.from_pretrained(model_ckpt)
        
        for param in self.f_reward_model.parameters():
            param.requires_grad = False
        self.f_reward_model.eval()
        
        # prepare spacy
        self.nlp = spacy.load("en_core_web_sm")
        self.sep = sep
        
        # the id corresponds to the sep token
        self.sep_id = self.reward_tokenizer.convert_tokens_to_ids(sep)

        # rewards
        self.factuality_positive_reward = factuality_positive_reward
        self.factuality_negative_reward = factuality_negative_reward

    def find_sep_position(self, input_ids):
        return torch.nonzero(input_ids == self.sep_id, as_tuple=False).squeeze().tolist()
    
    def process_one_generation(self, long_text, policy_text_len):
        
        sentence_end_char_idxs= split_text_to_sentences(long_text, self.nlp)
           
        sentences = [long_text[sentence_end_char_idxs[i]:sentence_end_char_idxs[i+1]] for i in range(len(sentence_end_char_idxs)-1)]
        
        # Initialize an empty list to store the end token indices of each sentence
        sentence_end_indices = []
    
        for sent_idx in range(len(sentences)):
            tokens = self.policy_tokenizer.tokenize(long_text[:sentence_end_char_idxs[sent_idx+1]])
            token_count = len(tokens)
            sentence_end_indices.append(token_count - 1)
        
        reward_sentences = [f"{sent} {self.sep}" for sent in sentences]
    
        reward_input = ' '.join(reward_sentences)
        
        # in case overlength    
        sentence_end_indices = [min(item, policy_text_len-1) for item in sentence_end_indices]
    
        return reward_input, sentence_end_indices
    
    def get_reward(self, 
                   prompts_input_ids: torch.tensor, 
                   prompts_attention_mask: torch.tensor, 
                   generated_input_ids: torch.tensor, # (B, output_len)
                   generated_attention_mask: torch.tensor, # (B, output_len)
                   generated_texts: List[str],
                   metadata=None,
                   ):
        
        batch_f_reward_inputs = []
        batch_sentence_end_indices = []
        
        # get the length of generated outputs
        policy_inputs_lens = torch.sum(generated_attention_mask, dim=1).tolist()

        for batch_idx, (meta, gen_text) in enumerate(zip(metadata, generated_texts)):
            reward_input, sentence_end_indices = self.process_one_generation(gen_text, 
                                                    policy_inputs_lens[batch_idx])

            # input for the factual reward model
            f_reward_input = f"{meta['prompt']} answer: {reward_input}"
            batch_f_reward_inputs.append(f_reward_input)
            
            # the indices of sentence ends for the policy model output
            batch_sentence_end_indices.append(sentence_end_indices)
        
        # get the reward
        with torch.no_grad():

            # to align with the token classification model
            inputs = self.reward_tokenizer([s.split() for s in batch_f_reward_inputs], 
                                           truncation=True, padding=True, 
                                           is_split_into_words=True,
                                           return_tensors="pt")
            inputs = inputs.to(self.f_reward_model.device)
            
            # factual reward model
            batch_f_pred = self.f_reward_model(**inputs)
            
        factuality_rewards = []
        n_corrects = []
        raw_rewards = []
        
        for text_idx, generated_text in enumerate(generated_texts):
            
            # extract the rewards from factual reward model output
            this_f_pred = batch_f_pred.logits[text_idx].detach().cpu()
            
            generated_text_input_ids = self.reward_tokenizer(
                batch_f_reward_inputs[text_idx].split(), 
                return_tensors="pt", 
                is_split_into_words=True,
                truncation=True).input_ids[0]
            
            # get the indices of </s>
            sep_indices = self.find_sep_position(generated_text_input_ids)
            sentence_f_reward_probs = this_f_pred[sep_indices]
            
            # align back to the original sentence
            policy_sentence_end_indices = batch_sentence_end_indices[text_idx]
            policy_inputs_len = policy_inputs_lens[text_idx]
            
            this_factuality_reward = [0]*policy_inputs_len
            this_raw_reward = []
            
            this_n_correct = 0
            
            for i, end_idx in enumerate(policy_sentence_end_indices):

                this_raw_reward.append(sentence_f_reward_probs[i].detach().cpu())
                
                # 0 is has error, 1 is no error
                f_error_type = torch.argmax(sentence_f_reward_probs[i][[0,2]]).item()
                factuality_reward = self.factuality_positive_reward if f_error_type == 1 else self.factuality_negative_reward
                
                # aggregate the rewards
                this_factuality_reward[end_idx] = factuality_reward
                
                if f_error_type == 1:
                    this_n_correct += 1
                    
            n_corrects.append(this_n_correct)
                
            factuality_rewards.append(this_factuality_reward)
            raw_rewards.append(this_raw_reward)
            
        return {"factuality_rewards": factuality_rewards,
                "n_sentences": [len(item) for item in batch_sentence_end_indices],
                "n_corrects": n_corrects,
                "raw_rewards": raw_rewards}

class FactualityCompactReward:
    def __init__(self,
                 tokenizer,
                 model_ckpt,
                 factuality_positive_reward = 1.0,
                 factuality_negative_reward = -1.0,
                 sep = "</s>",
                 ):
        
        # prepare policy tokenizer
        self.policy_tokenizer = tokenizer

        # prepare reward tokenizer
        self.reward_tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
        
        # prepare factual reward model
        self.f_reward_model = LongformerForTokenMultiClassification.from_pretrained(model_ckpt)
        
        for param in self.f_reward_model.parameters():
            param.requires_grad = False
        self.f_reward_model.eval()
        
        # prepare spacy
        self.nlp = spacy.load("en_core_web_sm")
        self.sep = sep
        
        # the id corresponds to the sep token
        self.sep_id = self.reward_tokenizer.convert_tokens_to_ids(sep)

        # rewards
        self.factuality_positive_reward = factuality_positive_reward
        self.factuality_negative_reward = factuality_negative_reward

    def find_sep_position(self, input_ids):
        return torch.nonzero(input_ids == self.sep_id, as_tuple=False).squeeze().tolist()
    
    def process_one_generation(self, long_text, policy_text_len):
        
        sentence_end_char_idxs= split_text_to_sentences(long_text, self.nlp)
           
        sentences = [long_text[sentence_end_char_idxs[i]:sentence_end_char_idxs[i+1]] for i in range(len(sentence_end_char_idxs)-1)]
        
        # Initialize an empty list to store the end token indices of each sentence
        sentence_end_indices = []
    
        for sent_idx in range(len(sentences)):
            tokens = self.policy_tokenizer.tokenize(long_text[:sentence_end_char_idxs[sent_idx+1]])
            token_count = len(tokens)
            sentence_end_indices.append(token_count - 1)
        
        reward_sentences = [f"{sent} {self.sep}" for sent in sentences]
    
        reward_input = ' '.join(reward_sentences)
        
        # in case overlength    
        sentence_end_indices = [min(item, policy_text_len-1) for item in sentence_end_indices]
    
        return reward_input, sentence_end_indices
    
    def get_reward(self, 
                   prompts_input_ids: torch.tensor, 
                   prompts_attention_mask: torch.tensor, 
                   generated_input_ids: torch.tensor, # (B, output_len)
                   generated_attention_mask: torch.tensor, # (B, output_len)
                   generated_texts: List[str],
                   metadata=None,
                   ):
        
        batch_f_reward_inputs = []
        batch_sentence_end_indices = []
        
        # get the length of generated outputs
        policy_inputs_lens = torch.sum(generated_attention_mask, dim=1).tolist()

        for batch_idx, (meta, gen_text) in enumerate(zip(metadata, generated_texts)):
            reward_input, sentence_end_indices = self.process_one_generation(gen_text, 
                                                    policy_inputs_lens[batch_idx])

            # input for the factual reward model
            f_reward_input = f"{meta['prompt']} answer: {reward_input}"
            batch_f_reward_inputs.append(f_reward_input)
            
            # the indices of sentence ends for the policy model output
            batch_sentence_end_indices.append(sentence_end_indices)
        
        # get the reward
        with torch.no_grad():

            # to align with the token classification model
            inputs = self.reward_tokenizer([s.split() for s in batch_f_reward_inputs], 
                                           truncation=True, padding=True, 
                                           is_split_into_words=True,
                                           return_tensors="pt")
            inputs = inputs.to(self.f_reward_model.device)
            
            # factual reward model
            batch_f_pred = self.f_reward_model(**inputs)
            
        factuality_rewards = []
        n_corrects = []
        raw_rewards = []
        
        for text_idx, generated_text in enumerate(generated_texts):
            
            # extract the rewards from factual reward model output
            this_f_pred = batch_f_pred.logits[text_idx].detach().cpu() # (# tokens, 6, 3)
            
            generated_text_input_ids = self.reward_tokenizer(
                batch_f_reward_inputs[text_idx].split(), 
                return_tensors="pt", 
                is_split_into_words=True,
                truncation=True).input_ids[0]
            
            # get the indices of </s>
            sep_indices = self.find_sep_position(generated_text_input_ids)
            sentence_f_reward_probs = this_f_pred[sep_indices]
            
            # align back to the original sentence
            policy_sentence_end_indices = batch_sentence_end_indices[text_idx]
            policy_inputs_len = policy_inputs_lens[text_idx]
            
            this_factuality_reward = [[0]*6]*policy_inputs_len
            this_raw_reward = []
            
            this_n_correct = 0
            
            for i, end_idx in enumerate(policy_sentence_end_indices):

                raw_reward = sentence_f_reward_probs[i].detach().cpu()
                raw_reward = raw_reward.view(-1, 3)
                this_raw_reward.append(raw_reward)
                
                # 0 is has error, 1 is no error
                f_error_types = torch.argmax(raw_reward[:, [0,2]], dim=1)
                factuality_reward = torch.where(f_error_types==1, self.factuality_positive_reward, self.factuality_negative_reward)
                
                # aggregate the rewards
                this_factuality_reward[end_idx] = factuality_reward.tolist()
                
                if torch.all(f_error_types == 1):
                    this_n_correct += 1
                    
            n_corrects.append(this_n_correct)
                
            factuality_rewards.append(this_factuality_reward)
            raw_rewards.append(this_raw_reward)
            
        return {"factuality_rewards": factuality_rewards,
                "n_sentences": [len(item) for item in batch_sentence_end_indices],
                "n_corrects": n_corrects,
                "raw_rewards": raw_rewards}

 
class FineGrainedCompactReward(BasicReward):
    
    def __init__(self,
                 tokenizer,
                 factual_model_ckpt,
                 kl_coef,
                 factuality_positive_reward = 1.0,
                 factuality_negative_reward = -1.0,
                 sep = "</s>"
                ):
        
        super().__init__(kl_coef)
        
        self.factuality_reward = FactualityCompactReward(tokenizer,
            factual_model_ckpt,
            factuality_positive_reward,
            factuality_negative_reward,
            sep = sep)
       
        self.nlp = spacy.load("en_core_web_sm")
    
    def get_finegrained_reward(self, prompts_input_ids, prompts_attention_mask, 
                            generated_input_ids, generated_attention_mask, 
                            generated_texts, metadata):
        
        fine_grained_rewards = []
        n_sub_sentences = []
        n_sentences = []
        
        factuality = self.factuality_reward.get_reward(prompts_input_ids, prompts_attention_mask, 
                                                    generated_input_ids, generated_attention_mask, 
                                                    generated_texts, metadata)

        n_sentences = factuality['n_sentences']
        factuality_rewards = factuality['factuality_rewards']
        n_factuality_correct = factuality['n_corrects']
            
        # combine the rewards
        for text_idx, generated_text in enumerate(generated_texts):
            
            fine_grained_reward = [np.sum(reward) for reward in factuality_rewards[text_idx]]
            
            fine_grained_rewards.append(fine_grained_reward)
            
        return {"rewards": fine_grained_rewards, 
                "n_sentences": n_sentences,
                "factuality_rewards": factuality_rewards,
                "n_factuality_correct": n_factuality_correct,
                }
        
        

    def get_reward(self, 
                   prompts_input_ids: torch.tensor, 
                   prompts_attention_mask: torch.tensor, 
                   generated_input_ids: torch.tensor, # (B, output_len)
                   generated_attention_mask: torch.tensor, # (B, output_len)
                   generated_texts: List[str],
                   metadata=None, 
                   ):
        
        rewards_output = self.get_finegrained_reward(prompts_input_ids, prompts_attention_mask, 
                            generated_input_ids, generated_attention_mask, 
                            generated_texts, metadata)
        
        return {'rewards/raw': rewards_output['rewards']}
            
        
    def eval_metrics(self, 
                prompts_input_ids: torch.tensor, 
                prompts_attention_mask: torch.tensor, 
                generated_input_ids: torch.tensor, # (B, output_len)
                generated_attention_mask: torch.tensor, # (B, output_len)
                generated_texts: List[str],
                metadata=None, 
                ):
        
        output = {}
        
        finegrained_rewards_output = self.get_finegrained_reward(prompts_input_ids, prompts_attention_mask, 
                            generated_input_ids, generated_attention_mask, 
                            generated_texts, metadata)
        
        # convert finegrained rewards to portions
        n_sentences = finegrained_rewards_output['n_sentences']
        
        factuality_ratios = []
        
        for text_idx, generated_text in enumerate(generated_texts):
            # factuality reward
            n_sentence = n_sentences[text_idx]
            n_factuality_correct = finegrained_rewards_output['n_factuality_correct'][text_idx]
            factuality_ratios.append(n_factuality_correct / n_sentence)
            
        # lens of generations
        generation_lens = torch.sum(generated_attention_mask, dim=-1).tolist()
        
        output.update({
            "eval/rewards": [np.sum(sublist) for sublist in finegrained_rewards_output['rewards']],
            "eval/factuality_ratios": factuality_ratios,
            "eval/n_sentences": n_sentences,
            "eval/lengths": generation_lens
        })
        
        return output
    
    
    def aggregate_metrics(self, wandb_table, value_columns):
        # how to average over the metrics in wandb table for reporting
        stats = {}
        for k in value_columns:
            stats[k] = np.mean([row[wandb_table.columns.index(k)] for row in wandb_table.data])
        
        # relevance ratios and factual ratios are weighted by the number of (sub)sentences
        
       
        stats['eval/factuality_ratios'] = (np.sum([row[wandb_table.columns.index('eval/factuality_ratios')]
                                                   * row[wandb_table.columns.index('eval/n_sentences')]
                                                   for row in wandb_table.data])
                                           / np.sum([row[wandb_table.columns.index('eval/n_sentences')]
                                                     for row in wandb_table.data]))
        
        return stats

class FineGrainedReward(BasicReward):
    
    def __init__(self,
                 tokenizer,
                 calculation_error_model_ckpt,
                 fabrication_model_ckpt,
                 context_inconsistency_model_ckpt,
                 factual_inconsistency_model_ckpt,
                 instruction_inconsistency_model_ckpt,
                 logical_inconsistency_model_ckpt,
                 kl_coef,
                 calculation_error_positive_reward = 1.0,
                 calculation_error_negative_reward = -1.0,
                 fabrication_positive_reward = 1.0,
                 fabrication_negative_reward = -1.0,
                 context_inconsistency_positive_reward = 1.0,
                 context_inconsistency_negative_reward = -1.0,
                 factual_inconsistency_positive_reward = 1.0,
                 factual_inconsistency_negative_reward = -1.0,
                 instruction_inconsistency_positive_reward = 1.0,
                 instruction_inconsistency_negative_reward = -1.0,
                 logical_inconsistency_positive_reward = 1.0,
                 logical_inconsistency_negative_reward = -1.0,
                 sep = "</s>"
                ):
        
        super().__init__(kl_coef)

        self.calculation_error_reward = FactualityReward(
            tokenizer,
            calculation_error_model_ckpt,
            calculation_error_positive_reward,
            calculation_error_negative_reward,
            sep = sep
        )

        self.fabrication_reward = FactualityReward(
            tokenizer,
            fabrication_model_ckpt,
            fabrication_positive_reward,
            fabrication_negative_reward,
            sep = sep
        )

        self.context_inconsistency_reward = FactualityReward(
            tokenizer,
            context_inconsistency_model_ckpt,
            context_inconsistency_positive_reward,
            context_inconsistency_negative_reward,
            sep = sep
        )

        self.factual_inconsistency_reward = FactualityReward(
            tokenizer,
            factual_inconsistency_model_ckpt,
            factual_inconsistency_positive_reward,
            factual_inconsistency_negative_reward,
            sep = sep
        )

        self.instruction_inconsistency_reward = FactualityReward(
            tokenizer,
            instruction_inconsistency_model_ckpt,
            instruction_inconsistency_positive_reward,
            instruction_inconsistency_negative_reward,
            sep = sep
        )

        self.logical_inconsistency_reward = FactualityReward(
            tokenizer,
            logical_inconsistency_model_ckpt,
            logical_inconsistency_positive_reward,
            logical_inconsistency_negative_reward,
            sep = sep
        )

        self.nlp = spacy.load("en_core_web_sm")
    
    def get_finegrained_reward(self, prompts_input_ids, prompts_attention_mask, 
                               generated_input_ids, generated_attention_mask, 
                               generated_texts, metadata):
 
        n_sentences = []
 
        calculation_error = self.calculation_error_reward.get_reward(
                prompts_input_ids, prompts_attention_mask, 
                generated_input_ids, generated_attention_mask, 
                generated_texts, metadata)

        fabrication = self.fabrication_reward.get_reward(
                prompts_input_ids, prompts_attention_mask, 
                generated_input_ids, generated_attention_mask, 
                generated_texts, metadata)

        context_inconsistency = self.context_inconsistency_reward.get_reward(
                prompts_input_ids, prompts_attention_mask, 
                generated_input_ids, generated_attention_mask, 
                generated_texts, metadata)

        factual_inconsistency = self.factual_inconsistency_reward.get_reward(
                prompts_input_ids, prompts_attention_mask, 
                generated_input_ids, generated_attention_mask, 
                generated_texts, metadata)

        instruction_inconsistency = self.instruction_inconsistency_reward.get_reward(
                prompts_input_ids, prompts_attention_mask, 
                generated_input_ids, generated_attention_mask, 
                generated_texts, metadata)

        logical_inconsistency = self.logical_inconsistency_reward.get_reward(
                prompts_input_ids, prompts_attention_mask, 
                generated_input_ids, generated_attention_mask, 
                generated_texts, metadata)

        n_sentences = calculation_error['n_sentences']

        calculation_error_rewards = calculation_error['factuality_rewards']
        fabrication_rewards = fabrication['factuality_rewards']
        context_inconsistency_rewards = context_inconsistency['factuality_rewards']
        factual_inconsistency_rewards = factual_inconsistency['factuality_rewards']
        instruction_inconsistency_rewards = instruction_inconsistency['factuality_rewards']
        logical_inconsistency_rewards = logical_inconsistency['factuality_rewards']

        n_calculation_error_correct = calculation_error['n_corrects']
        n_fabrication_correct = fabrication['n_corrects']
        n_context_inconsistency_correct = context_inconsistency['n_corrects']
        n_factual_inconsistency_correct = factual_inconsistency['n_corrects']
        n_instruction_inconsistency_correct = instruction_inconsistency['n_corrects']
        n_logical_inconsistency_correct = logical_inconsistency['n_corrects']

        # combine the rewards
        fine_grained_rewards = []
        for text_idx, generated_text in enumerate(generated_texts):
 
            fine_grained_reward = [a+b+c+d+e+f for a,b,c,d,e,f in zip(
                calculation_error_rewards[text_idx],
                fabrication_rewards[text_idx],
                context_inconsistency_rewards[text_idx],
                factual_inconsistency_rewards[text_idx],
                instruction_inconsistency_rewards[text_idx],
                logical_inconsistency_rewards[text_idx],
            )]
 
            fine_grained_rewards.append(fine_grained_reward)
 
        return {"rewards": fine_grained_rewards, 
                "n_sentences": n_sentences,
                "calculation_error_rewards": calculation_error_rewards,
                "fabrication_rewards": fabrication_rewards,
                "context_inconsistency_rewards": context_inconsistency_rewards,
                "factual_inconsistency_rewards": factual_inconsistency_rewards,
                "instruction_inconsistency_rewards": instruction_inconsistency_rewards,
                "logical_inconsistency_rewards": logical_inconsistency_rewards,
                "n_calculation_error_correct": n_calculation_error_correct,
                "n_fabrication_correct": n_fabrication_correct,
                "n_context_inconsistency_correct": n_context_inconsistency_correct,
                "n_factual_inconsistency_correct": n_factual_inconsistency_correct,
                "n_instruction_inconsistency_correct": n_instruction_inconsistency_correct,
                "n_logical_inconsistency_correct": n_logical_inconsistency_correct,
                }

    def get_reward(self, 
                   prompts_input_ids: torch.tensor, 
                   prompts_attention_mask: torch.tensor, 
                   generated_input_ids: torch.tensor, # (B, output_len)
                   generated_attention_mask: torch.tensor, # (B, output_len)
                   generated_texts: List[str],
                   metadata=None, 
                   ):
        
        rewards_output = self.get_finegrained_reward(
                prompts_input_ids, prompts_attention_mask, 
                generated_input_ids, generated_attention_mask, 
                generated_texts, metadata
        )
        
        return {'rewards/raw': rewards_output['rewards']}
            
        
    def eval_metrics(self, 
                prompts_input_ids: torch.tensor, 
                prompts_attention_mask: torch.tensor, 
                generated_input_ids: torch.tensor, # (B, output_len)
                generated_attention_mask: torch.tensor, # (B, output_len)
                generated_texts: List[str],
                metadata=None, 
                ):
        
        output = {}
        
        finegrained_rewards_output = self.get_finegrained_reward(
                prompts_input_ids, prompts_attention_mask, 
                generated_input_ids, generated_attention_mask, 
                generated_texts, metadata
        )
        
        # convert finegrained rewards to portions
        n_sentences = finegrained_rewards_output['n_sentences']
 
        calculation_error_ratios = []
        fabrication_ratios = []
        context_inconsistency_ratios = []
        factual_inconsistency_ratios = []
        instruction_inconsistency_ratios = []
        logical_inconsistency_ratios = []
 
        for text_idx, generated_text in enumerate(generated_texts):
            n_sentence = n_sentences[text_idx]

            n_calculation_error_correct = finegrained_rewards_output['n_calculation_error_correct'][text_idx]
            calculation_error_ratios.append(n_calculation_error_correct / n_sentence)
       
            n_fabrication_correct = finegrained_rewards_output['n_fabrication_correct'][text_idx]
            fabrication_ratios.append(n_fabrication_correct / n_sentence)
       
            n_context_inconsistency_correct = finegrained_rewards_output['n_context_inconsistency_correct'][text_idx]
            context_inconsistency_ratios.append(n_context_inconsistency_correct / n_sentence)
       
            n_factual_inconsistency_correct = finegrained_rewards_output['n_factual_inconsistency_correct'][text_idx]
            factual_inconsistency_ratios.append(n_factual_inconsistency_correct / n_sentence)
       
            n_instruction_inconsistency_correct = finegrained_rewards_output['n_instruction_inconsistency_correct'][text_idx]
            instruction_inconsistency_ratios.append(n_instruction_inconsistency_correct / n_sentence)
       
            n_logical_inconsistency_correct = finegrained_rewards_output['n_logical_inconsistency_correct'][text_idx]
            logical_inconsistency_ratios.append(n_logical_inconsistency_correct / n_sentence)
       
        # compute rouge scores
        rouge_scores = get_rouge_scores(generated_texts, [m['references'] for m in metadata])
        
        # lens of generations
        generation_lens = torch.sum(generated_attention_mask, dim=-1).tolist()
        
        output.update({
            "eval/rouge": rouge_scores,
            "eval/rewards": [np.sum(sublist) for sublist in finegrained_rewards_output['rewards']],
            "eval/calculation_error_ratios": calculation_error_ratios,
            "eval/fabrication_ratios": fabrication_ratios,
            "eval/context_inconsistency_ratios": context_inconsistency_ratios,
            "eval/factual_inconsistency_ratios": factual_inconsistency_ratios,
            "eval/instruction_inconsistency_ratios": instruction_inconsistency_ratios,
            "eval/logical_inconsistency_ratios": logical_inconsistency_ratios,
            "eval/n_sentences": n_sentences,
            "eval/lengths": generation_lens
        })
        
        return output
    
    
    def aggregate_metrics(self, wandb_table, value_columns):
        # how to average over the metrics in wandb table for reporting
        stats = {}
        for k in value_columns:
            stats[k] = np.mean([row[wandb_table.columns.index(k)] for row in wandb_table.data])
        
        # relevance ratios and factual ratios are weighted by the number of (sub)sentences
        
        stats['eval/calculation_error_ratios'] = (
                np.sum([row[wandb_table.columns.index('eval/calculation_error_ratios')]
                        * row[wandb_table.columns.index('eval/n_sentences')]
                        for row in wandb_table.data])
                / np.sum([row[wandb_table.columns.index('eval/n_sentences')]
                          for row in wandb_table.data])
        )

        stats['eval/fabrication_ratios'] = (
                np.sum([row[wandb_table.columns.index('eval/fabrication_ratios')]
                        * row[wandb_table.columns.index('eval/n_sentences')]
                        for row in wandb_table.data])
                / np.sum([row[wandb_table.columns.index('eval/n_sentences')]
                          for row in wandb_table.data])
        )

        stats['eval/context_inconsistency_ratios'] = (
                np.sum([row[wandb_table.columns.index('eval/context_inconsistency_ratios')]
                        * row[wandb_table.columns.index('eval/n_sentences')]
                        for row in wandb_table.data])
                / np.sum([row[wandb_table.columns.index('eval/n_sentences')]
                          for row in wandb_table.data])
        )

        stats['eval/factual_inconsistency_ratios'] = (
                np.sum([row[wandb_table.columns.index('eval/factual_inconsistency_ratios')]
                        * row[wandb_table.columns.index('eval/n_sentences')]
                        for row in wandb_table.data])
                / np.sum([row[wandb_table.columns.index('eval/n_sentences')]
                          for row in wandb_table.data])
        )

        stats['eval/instruction_inconsistency_ratios'] = (
                np.sum([row[wandb_table.columns.index('eval/instruction_inconsistency_ratios')]
                        * row[wandb_table.columns.index('eval/n_sentences')]
                        for row in wandb_table.data])
                / np.sum([row[wandb_table.columns.index('eval/n_sentences')]
                          for row in wandb_table.data])
        )

        stats['eval/logical_inconsistency_ratios'] = (
                np.sum([row[wandb_table.columns.index('eval/logical_inconsistency_ratios')]
                        * row[wandb_table.columns.index('eval/n_sentences')]
                        for row in wandb_table.data])
                / np.sum([row[wandb_table.columns.index('eval/n_sentences')]
                          for row in wandb_table.data])
        )

        return stats

