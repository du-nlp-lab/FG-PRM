# *********************************************************
#  Version 1
#  Author: Yushi Hu
#  Date: 2023-06-20
#  Description: the helper functions that split long text into sentences and subsentences
#  All Rights Reserved.
#  *********************************************************

import re

# split long text into sentences
def split_text_to_sentences(long_text, spacy_nlp):
    """ Split long text into sub-sentences ONLY based on SpaCy

    Args:
        long_text(str): original text sequence
        spacy_nlp(nlp): a nlp class in SpaCy package

    Returns:
        char_starts(list of int): list of indeces indicating the split indexes for sub-sentences in the long_text.
    """
    ### PRM ###
    ori_long_text = long_text

    answer_step = None
    h = 'Step'
    match = re.search('(The answer is:.*)$', long_text, re.M)
    if match is not None:
        raw_steps = long_text[:match.span(1)[0]]
        raw_steps = [c for c in raw_steps.split(h) if c]
        answer_step = match.group(1)
        raw_steps.append(answer_step)

    if answer_step is None:
        match = re.search('(# Answer\n*.*)$', long_text, re.M)
        if match is not None:
            raw_steps = long_text[:match.span(1)[0]]
            raw_steps = [c for c in raw_steps.split(h) if c]
            answer_step = match.group(1)
            raw_steps.append(answer_step)

    if answer_step is None:
        raw_steps = long_text.split('Step')

    steps = []
    for s_idx, s in enumerate(raw_steps):
        # the text before "Step 1"
        if 0 == s_idx:
            if '' == s.strip():
                end_ids = [0]
            elif '1' == s.strip()[0]:
                steps.append('Step' + s)
                end_ids = [0]
            else:
                # if any texts are before "Step 1", we ignore them
                steps.append(s)
                end_ids = [len(s)]
            continue

        # the last answer step
        if len(raw_steps) == s_idx + 1 and answer_step is not None:
            steps.append(s)
            continue

        # all intermediate stpes
        if '' != s.strip():
            steps.append('Step' + s)

    for text in steps:
        if 0 < len(text) and 'Step' == text[:4]:
            end_ids.append(min(end_ids[-1]+len(text), len(ori_long_text)))

#    ### ORM ###
#    end_ids = [0, len(long_text)]

    return end_ids

    ### ORIGIN ###
#    doc = spacy_nlp(long_text)
#    return [0] + [sent.end_char for sent in doc.sents if len(str(sent).strip()) > 0]
    
    
# split long text into subsentences
def split_text_to_subsentences(long_text, spacy_nlp):
    """ Split long text into sub-sentences based on SpaCy Sentencizer and fine-grained by .;!?

    Args:
        long_text(str): original text sequence
        spacy_nlp(nlp): a nlp class in SpaCy package

    Returns:
        char_starts(list of int): list of indeces indicating the split indexes for sub-sentences in the long_text.
    """

    def get_sub_sentence_starts(tokens, min_subsent_words=5):
        """ Check SpaCy split sentences by .;!? and split it again by .;!? if necessary

        Args:
            tokens(list of str): token list
            min_subsent_words(int): minimum length of a sub-sentence

        Returns:
            is_subsent_starts(list of bool): list of bool indicating whether a token starts a
                                             sub-sentence. The length is equal to the "tokens" argument.
        """

        def _is_tok_end_of_subsent(tok):
            """ End token or not
                If a token is in [.;!?], it is an end token
            """
            if re.match('[,;!?]', tok[-1]) is not None:
                return True
            return False

        # assert len(tokens) > 0
        is_subsent_starts = [True]
        prev_tok = tokens[0]
        prev_subsent_start_idx = 0
        for i, tok in enumerate(tokens[1:]):
            tok_id = i + 1
            if _is_tok_end_of_subsent(prev_tok) and tok_id + min_subsent_words < len(tokens):
                # find a possible end token
                if tok_id - prev_subsent_start_idx < min_subsent_words:
                    # length of new sub-sentence doesn't satisfy the requirement
                    if prev_subsent_start_idx > 0:
                        # remove the last sub-sentence and set a new one from the next token
                        is_subsent_starts += [True]
                        is_subsent_starts[prev_subsent_start_idx] = False
                        prev_subsent_start_idx = tok_id
                    else:
                        # ignore if it is just next to the start of the current token sequence
                        is_subsent_starts += [False]
                else:
                    # normal new sub-sentence since its length is larger than min_subsent_words
                    is_subsent_starts += [True]
                    prev_subsent_start_idx = tok_id
            else:
                # normal in-sentence token
                is_subsent_starts += [False]
            prev_tok = tok

        return is_subsent_starts


    def tokenize_with_indices(text):
        """ Extract index for each token in the current text
            Indeces are based on the current text.
            Tokens are split based on **white space**.

        Args:
            text(str): input text

        Returns:
            token_indices(pair<str, int>): pairs of (token, index)
        """
        tokens = text.split()
        token_indices = []

        current_index = 0
        for token in tokens:
            start_index = text.find(token, current_index)
            token_indices.append((token, start_index))
            current_index = start_index + len(token)

        return token_indices
    
    doc = spacy_nlp(long_text) # split long_text to sub-sentences by SpaCy
    sentence_start_char_idxs= [0] + [sent.end_char for sent in doc.sents if len(str(sent).strip()) > 0] # record coarse sentence split index, fine-grained by .;?! at line 113
    
    char_starts = []
    
    for sentence_idx, sentence_start_char_idx in enumerate(sentence_start_char_idxs[:-1]):
        
        sentence = long_text[sentence_start_char_idx: sentence_start_char_idxs[sentence_idx+1]] # extract setences from the original text
        
        tokens_with_indices = tokenize_with_indices(sentence) # get start indexes of each token based on the current sentence
        
        tokens = [i[0] for i in tokens_with_indices] # extract tokens split by white space
        is_sub_starts = get_sub_sentence_starts(tokens, min_subsent_words=5) # fine-grain SpaCy splits by .;!? and assign start flag for each token
        
        # record real sentence split index by is_sub_starts compared to line 102
        for token_with_idx, is_sub_start in zip(tokens_with_indices, is_sub_starts):
            if is_sub_start:
                char_starts.append(sentence_start_char_idx + token_with_idx[1])
    
    return char_starts + [len(long_text)]
