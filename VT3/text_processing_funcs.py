import sys
import json
import progressbar
import numpy as np
from nltk import word_tokenize

import nltk
import re
from difflib import SequenceMatcher
from nltk.corpus import stopwords
stopword_set = set(stopwords.words('english'))

def clean_one_token(token):
    char_list = []
    for c in token:
        if c == '(':
            one_c = ''
        elif c == ')':
            one_c = ''
        else:
            one_c = c
        char_list.append(one_c)
    return ''.join(char_list)

def clean_text(text):
    # remove unncessary trailing white spaces and remove
    # paranthesis. eg "... (1985)" -> "... 1985"
    res_list = []
    token_list = text.strip().split()
    for token in token_list:
        res_list.append(clean_one_token(token))
    return ' '.join(res_list).strip()

def map_to_pure_content(content_sequence):
    # get list of all found slot occurances in the reference sentence
    # ex: arg content_sequence -> "Who let the __Athlete__4__ out in 1984 when __Data__1__ "
    # map_to_pure_content() -> "__Athlete__4__ __Data__1__"
    # if no slot occurances in reference sentence then map_to_pure_content() -> '' (empty string)
    res_list = []
    for token in content_sequence.split():
        if token.startswith('__') and token.endswith('__'):
            res_list.append(token)
        else:
            pass
    return ' '.join(res_list).strip()

def transform_matching_string(match_string):
    special_char_set = set(list(r"!@#$%^&*()[]{};:,./<>?\|`~-=_+"))
    res_str = ''
    for one_char in match_string:
        if one_char in special_char_set:
            one_char = r"\\" + one_char
        else:
            one_char = one_char
        res_str += one_char
    return res_str

def return_valid_length(substring):
    '''
        return number of valid tokens exist in the matched string
        if the matched string only contains stopword or the overlapped 
        length is too small, then we reject the replacement
    '''
    token_list = substring.strip().split()
    valid_len = 0
    for token in token_list:
        if token.lower() in stopword_set:
            pass
        else:
            valid_len += 1
    #if len(substring) < 3: # single letter matching
    if len(substring) < 3 and substring.isalpha():
        valid_len = 0
    elif len(substring) < 2:
        valid_len = 0
    else:
        pass
    return valid_len

def find_longest_common_substring(string1, string2):
    match = SequenceMatcher(None, string1, string2).find_longest_match(0, len(string1), 0, len(string2))
    match_string = string1[match.a: match.a + match.size].strip()
    valid_len = return_valid_length(match_string)
    match_span = match.size
    return match_string, valid_len, match_span

    # case 3816 or 3846

def find_final_substring(tokenized_reference, slot_value_text):
    match_1, valid_len_1, span_1 = find_longest_common_substring(tokenized_reference, slot_value_text)
    match_2, valid_len_2, span_2 = find_longest_common_substring(slot_value_text, tokenized_reference)
    if valid_len_1 > valid_len_2:
        return match_1, valid_len_1, span_1
    else:
        return match_2, valid_len_2, span_2

def check_result(text):
    flag = True
    token_list = text.strip().split()
    for token in token_list:
        if token.startswith('__') and token.endswith('__'):
            try:
                assert len(token.strip('__').split('__')) == 2
            except:
                flag = False
                break
    return flag

def replace_reference(tokenized_reference, slot_key_text, slot_key_position, slot_value_text):
    cache_result = tokenized_reference
    match_string, valid_len, _ = find_final_substring(tokenized_reference, slot_value_text)
    match_span = len(match_string)
    #print (match_string, len(match_string))
    #print (valid_len, match_span)
    if valid_len == 0:
        return tokenized_reference
    reference_len = len(tokenized_reference)
    flag = False
    for k in range(reference_len):
        curr_span_string = tokenized_reference[k:k+match_span]
        if curr_span_string == match_string:
            flag = True
            first_part = tokenized_reference[:k] # part before matched string
            second_part = tokenized_reference[k+match_span:] # part after matched string
            middle_part = ' ' + slot_key_text + str(slot_key_position) + '__' + ' '
            # example: (' ' + __Athlete__ + str(4) + '__' + ' ') => (" __Athlete__4__ ")
            res_str = first_part + middle_part + second_part # replaces matched string in reference sentence with above representation
            # question: we don't handle the case where the key is repeated more than once in reference sentence?
            break 
        else:
            continue
    assert flag == True # ensure the matching string found has been replaced by new slot key representation
    if check_result(res_str): # check if the replaced representation ie __xyz_3__ is present in the sentence, returns True if yes, False if not
        return res_str
    else:
        return cache_result

def process_one_instance(tokenized_reference, ordered_cell_dict):
    res_text = clean_text(tokenized_reference) # remove trailing white spaces & parenthesis in sentence
    for idx in range(len(ordered_cell_dict)): # loop over all slot keys (headers)
        item = ordered_cell_dict[idx] # item -> {"slot_key": '__x__', "slot_value": ''}
        one_slot_key_text = item['slot_key']
        #if one_slot_key_text == '__section_title__':
        #    continue
        one_slot_value_text = clean_text(item['slot_value'])
        one_slot_key_position = idx
        res_text = replace_reference(res_text, one_slot_key_text, one_slot_key_position, one_slot_value_text)
        # replace_reference -> check if slot value is present in the reference sentence. if match is present 
        # then replace matching part with '__key__position__' (eg __Athlete__4__) in the sentence
    # after loop: we should have a ref sentence where all slot occurances are replaced by their new representation
    # and non-matching parts remain the same

    # map_to_pure_content returns '__slotkey__3__ __xyz__1__ __abc__5__' if they're found in res_text
    return map_to_pure_content(res_text) 
