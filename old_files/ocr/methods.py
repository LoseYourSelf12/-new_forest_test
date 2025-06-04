import re
from Levenshtein import distance
alphabet = 'abcdefghijklmnopqrstuvwxyz'

def long_phrase(levdist, phrase, threshold=3):
    # parse phrase to words
    phrase = phrase.split()
    
    # find phrase with min distanse
    min_item = ['',100]
    for item in levdist:
        if item[1] < min_item[1] and len(item[0].split()) == len(phrase):
            min_item = item
    
    # encode phrase in 1 word
    result_zip = ''
    for i,word in enumerate(min_item[0].split()):
        if distance(phrase[i],word) < 1 + len(phrase[i])//3:
            result_zip += '{}'.format(alphabet[i])
        else:
            result_zip += '{}'.format(alphabet[-(i+1)])
        phrase[i] = '{}'.format(alphabet[i])
    phrase = ''.join(phrase)
    return min_item, phrase, result_zip

def fuzzy_check(phrase, text):
    phrase_len = len(phrase.split())

    text = text.split()
    text_len = len(text)

    text_set = [' '.join(text[i : i+phrase_len]) for i in range(text_len-phrase_len+1)]

    levdist = [[text_window, distance(phrase, text_window)] for text_window in text_set]
    
    threshold = 1 + len(phrase)//3

    result = [[txt,rt] for txt,rt in levdist if rt < threshold]

    if len(result)<1 and len(set(phrase.split()))>1:

        min_item, phrase, phrase_zip = long_phrase(levdist, phrase)
        
        levdist = [[min_item[0], distance(phrase, phrase_zip)]]
        result  = [[txt,rt] for txt,rt in levdist if rt < 1]
    if result: result = True
    else: result = False
    return result

def reg_check(phrase, text):
    pattern_phrase = ''
    for word in phrase.split():
        pattern_phrase += '\s' + word + '\s'
    pattern = rf'{pattern_phrase}'

    matches = re.search(pattern, text)
    if matches: matches = True
    else: matches = False
    return matches
