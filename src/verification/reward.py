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

from verification.my_longformer import LongformerForSequenceClassification, LongformerForTokenClassification, LongformerForTokenMultiClassification

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
                this_factuality_reward[end_idx] = factuality_reward
                
                if torch.all(f_error_types == 1):
                    this_n_correct += 1
                    
            n_corrects.append(this_n_correct)
                
            factuality_rewards.append(this_factuality_reward)
            raw_rewards.append(this_raw_reward)
            
        return {"factuality_rewards": factuality_rewards,
                "n_sentences": [len(item) for item in batch_sentence_end_indices],
                "n_corrects": n_corrects,
                "raw_rewards": raw_rewards}

