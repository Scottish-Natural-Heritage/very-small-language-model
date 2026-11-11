# SPDX-FileCopyrightText: 2026 Mike Coats <mike.coats@nature.scot>
# SPDX-License-Identifier: MIT

"""Shared n-gram language model building and text generation code.

This module is the single place where the model-building (``build_*_model``)
and text-generation (``next_word``, ``generate_sentence_*``, etc.) logic
lives, so it can be reused from the notebooks (``03``-``05``) instead of being
duplicated in each of them.
"""

import pickle
from itertools import accumulate
from random import randint

# Sensible ranges derived from analysing our 6 source blogs' RSS feeds (10th-90th percentile),
# used below instead of hard-coded values when generating text.
MIN_WORDS_PER_SENTENCE = 3
MAX_WORDS_PER_SENTENCE = 36
MIN_SENTENCES_PER_PARAGRAPH = 1
MAX_SENTENCES_PER_PARAGRAPH = 4

# Each generated sentence is assembled from two halves (backward + forward from the prompt),
# so the per-half word cap should be half of the full-sentence range above, otherwise a full
# sentence could end up roughly double the realistic length.
HALF_MIN_WORDS_PER_SENTENCE = max(1, MIN_WORDS_PER_SENTENCE // 2)
HALF_MAX_WORDS_PER_SENTENCE = max(HALF_MIN_WORDS_PER_SENTENCE, MAX_WORDS_PER_SENTENCE // 2)


def upsert(into_model, from_token, to_token):

    # Not seen our "from" before? Create the entry.
    if from_token not in into_model:
        into_model[from_token] = {to_token: 1}
        return

    # Seen the "from", but not the "to"? Add this as an option.
    if to_token not in into_model[from_token]:
        into_model[from_token][to_token] = 1
    # Seen the "from" and the "to"? Add weight to this option.
    else:
        into_model[from_token][to_token] += 1


def build_forward1_model(sentences_words):
    model = {}
    # Loop over every sentence in every blog.
    for sentence in sentences_words:
        tokens = ['START'] + sentence + ['END']

        # For each word "from" & "to" pair, add them to the model.
        for idx in range(0, len(tokens) - 1):
            token_prev = tokens[idx]
            token_this = tokens[idx + 1]
            upsert(model, token_prev, token_this)

    # Save the model for re-use.
    return model


def build_backward1_model(sentences_words):
    model = {}
    for sentence in sentences_words:
        tokens = ['END'] + list(reversed(sentence)) + ['START']

        for idx in range(len(tokens) - 1):
            token_prev = tokens[idx]
            token_this = tokens[idx + 1]
            upsert(model, token_prev, token_this)

    return model


def build_forward2_model(sentences_words):
    model = {}
    # Loop over every sentence in every blog.
    for sentence in sentences_words:
        tokens = (
            [
                'START',
                'START',
            ]
            + sentence
            + ['END', 'END']
        )
        # For each word "from, from" & "to" pair, add them to the model.
        for idx in range(0, len(tokens) - 2):
            token_prev = ' '.join([tokens[idx], tokens[idx + 1]])
            token_this = tokens[idx + 2]
            upsert(model, token_prev, token_this)
    return model


def build_backward2_model(sentences_words):
    model = {}
    for sentence in sentences_words:
        tokens = (
            [
                'END',
                'END',
            ]
            + list(reversed(sentence))
            + ['START', 'START']
        )
        # For each word "from, from" & "to" pair, add them to the model.
        for idx in range(0, len(tokens) - 2):
            token_prev = ' '.join([tokens[idx], tokens[idx + 1]])
            token_this = tokens[idx + 2]
            upsert(model, token_prev, token_this)
    return model


def build_forward3_model(sentences_words):
    model = {}
    # Loop over every sentence in every blog.
    for sentence in sentences_words:
        tokens = (
            [
                'START',
                'START',
                'START',
            ]
            + sentence
            + ['END', 'END', 'END']
        )
        # For each word "from, from, from" & "to" pair, add them to the model.
        for idx in range(0, len(tokens) - 3):
            token_prev = ' '.join([tokens[idx], tokens[idx + 1], tokens[idx + 2]])
            token_this = tokens[idx + 3]
            upsert(model, token_prev, token_this)
    return model


def build_backward3_model(sentences_words):
    model = {}
    for sentence in sentences_words:
        tokens = (
            [
                'END',
                'END',
                'END',
            ]
            + list(reversed(sentence))
            + ['START', 'START', 'START']
        )
        # For each word "from, from, from" & "to" pair, add them to the model.
        for idx in range(0, len(tokens) - 3):
            token_prev = ' '.join([tokens[idx], tokens[idx + 1], tokens[idx + 2]])
            token_this = tokens[idx + 3]
            upsert(model, token_prev, token_this)
    return model


def build_words1_model(sentences_words):
    model = {}
    for sentence in sentences_words:
        for word in sentence:
            if word not in model:
                model[word] = 1
            else:
                model[word] += 1
    return model


def build_words2_model(sentences_words):
    model = {}
    for sentence in sentences_words:
        for idx in range(0, len(sentence) - 1):
            pair = ' '.join([sentence[idx], sentence[idx + 1]])
            if pair not in model:
                model[pair] = 1
            else:
                model[pair] += 1
    return model


def build_words3_model(sentences_words):
    model = {}
    for sentence in sentences_words:
        for idx in range(0, len(sentence) - 2):
            triple = ' '.join([sentence[idx], sentence[idx + 1], sentence[idx + 2]])
            if triple not in model:
                model[triple] = 1
            else:
                model[triple] += 1
    return model


def next_word(in_model, this_word):
    # Find all the "to" words for our "from".
    candidates = in_model[this_word].items()
    words = [v[0] for v in candidates]
    scores = [v[1] for v in candidates]

    # Score each "to" word based on its weight, and line it up.
    # e.g 1 = A, 2-6 = B, 7-9 = C, etc.
    upper_lim = sum(scores)
    accumulated_scores = accumulate(scores)
    choices = zip(accumulated_scores, words, strict=True)

    # Roll a die based on the maximum number.
    die_roll = randint(0, upper_lim)

    # Pick the one that matches our dice roll.
    chosen = list(filter(lambda c: c[0] >= die_roll, choices))[0][1]
    return chosen


def generate_sentence_forward1(in_model, seed_word):
    # Generate sentences until a natural end, with a word-count cap sampled from half the
    # sensible words-per-sentence range (this is only one half of the sentence - the other
    # half is generated independently going the other direction from the prompt).
    last_word = seed_word
    sentence = []
    for _ in range(randint(HALF_MIN_WORDS_PER_SENTENCE, HALF_MAX_WORDS_PER_SENTENCE)):
        word = next_word(in_model, last_word)
        last_word = word
        if last_word == 'END':
            break
        sentence.append(word)

    return sentence


def generate_sentence_backward1(in_model, seed_word):
    # Generate sentences until a natural end, with a word-count cap sampled from half the
    # sensible words-per-sentence range (this is only one half of the sentence - the other
    # half is generated independently going the other direction from the prompt).
    last_word = seed_word
    sentence = []
    for _ in range(randint(HALF_MIN_WORDS_PER_SENTENCE, HALF_MAX_WORDS_PER_SENTENCE)):
        word = next_word(in_model, last_word)
        last_word = word
        if last_word == 'START':
            break
        sentence.append(word)

    return sentence


def generate_sentence_forward2(in_bigram_model, in_unigram_model, seed_word):
    # Use the unigram model for a single word forecast, turning our one word prompt into two.
    second_word = next_word(in_unigram_model, seed_word)
    sentence = []
    if second_word == 'END':
        return sentence
    sentence.append(second_word)

    # Generate sentences until a natural end, with a word-count cap sampled from half the
    # sensible words-per-sentence range (this is only one half of the sentence - the other
    # half is generated independently going the other direction from the prompt).
    last_pair = f'{seed_word} {second_word}'
    for _ in range(randint(HALF_MIN_WORDS_PER_SENTENCE, HALF_MAX_WORDS_PER_SENTENCE)):
        word = next_word(in_bigram_model, last_pair)
        if word == 'END':
            break
        sentence.append(word)
        last_pair = f'{last_pair.split(" ")[1]} {word}'

    return sentence


def generate_sentence_backward2(in_bigram_model, in_unigram_model, seed_word):
    # Use the unigram model for a single word forecast, turning our one word prompt into two.
    second_word = next_word(in_unigram_model, seed_word)
    sentence = []
    if second_word == 'START':
        return sentence
    sentence.append(second_word)

    # Generate sentences until a natural end, with a word-count cap sampled from half the
    # sensible words-per-sentence range (this is only one half of the sentence - the other
    # half is generated independently going the other direction from the prompt).
    last_pair = f'{seed_word} {second_word}'
    for _ in range(randint(HALF_MIN_WORDS_PER_SENTENCE, HALF_MAX_WORDS_PER_SENTENCE)):
        word = next_word(in_bigram_model, last_pair)
        if word == 'START':
            break
        sentence.append(word)
        last_pair = f'{last_pair.split(" ")[1]} {word}'

    return sentence


def generate_sentence_forward3(in_trigram_model, in_bigram_model, in_unigram_model, seed_word):
    # Use the unigram model for a single word forecast, turning our one word prompt into two.
    second_word = next_word(in_unigram_model, seed_word)
    sentence = []
    if second_word == 'END':
        return sentence
    sentence.append(second_word)

    # Use the bigram model for a single word forecast, turning our two word prompt into three.
    third_word = next_word(in_bigram_model, f'{seed_word} {second_word}')
    if third_word == 'END':
        return sentence
    sentence.append(third_word)

    # Generate sentences until a natural end, with a word-count cap sampled from half the
    # sensible words-per-sentence range (this is only one half of the sentence - the other
    # half is generated independently going the other direction from the prompt).
    last_triple = f'{seed_word} {second_word} {third_word}'
    for _ in range(randint(HALF_MIN_WORDS_PER_SENTENCE, HALF_MAX_WORDS_PER_SENTENCE)):
        word = next_word(in_trigram_model, last_triple)
        if word == 'END':
            break
        sentence.append(word)
        triple_words = last_triple.split(' ')
        last_triple = f'{triple_words[1]} {triple_words[2]} {word}'

    return sentence


def generate_sentence_backward3(in_trigram_model, in_bigram_model, in_unigram_model, seed_word):
    # Use the unigram model for a single word forecast, turning our one word prompt into two.
    second_word = next_word(in_unigram_model, seed_word)
    sentence = []
    if second_word == 'START':
        return sentence
    sentence.append(second_word)

    # Use the bigram model for a single word forecast, turning our two word prompt into three.
    third_word = next_word(in_bigram_model, f'{seed_word} {second_word}')
    if third_word == 'START':
        return sentence
    sentence.append(third_word)

    # Generate sentences until a natural end, with a word-count cap sampled from half the
    # sensible words-per-sentence range (this is only one half of the sentence - the other
    # half is generated independently going the other direction from the prompt).
    last_triple = f'{seed_word} {second_word} {third_word}'
    for _ in range(randint(HALF_MIN_WORDS_PER_SENTENCE, HALF_MAX_WORDS_PER_SENTENCE)):
        word = next_word(in_trigram_model, last_triple)
        if word == 'START':
            break
        sentence.append(word)
        triple_words = last_triple.split(' ')
        last_triple = f'{triple_words[1]} {triple_words[2]} {word}'

    return sentence


def generate_trigram_sentence(
    seed_word, forward3, forward2, forward1, backward3, backward2, backward1
):
    # Build one full sentence (as a list of words) using the trigram models.
    generated_start = list(
        reversed(generate_sentence_backward3(backward3, backward2, backward1, seed_word))
    )
    generated_end = generate_sentence_forward3(forward3, forward2, forward1, seed_word)
    return generated_start + [seed_word] + generated_end


def least_likely_trigram(sentence_words, words3_model):
    # Find the three-word set in the sentence with the lowest count in the words_3 model,
    # while still requiring it to have occurred at least once in the corpus.
    worst_triple = None
    worst_count = None
    for idx in range(0, len(sentence_words) - 2):
        triple = (sentence_words[idx], sentence_words[idx + 1], sentence_words[idx + 2])
        count = words3_model.get(f'{triple[0]} {triple[1]} {triple[2]}', 0)
        if count > 0 and (worst_count is None or count < worst_count):
            worst_count = count
            worst_triple = triple
    return worst_triple


def next_seed_word(sentence_words, topic, words3_model):
    # Pick one of the three words from the least likely (but still pre-occurring) trigram,
    # or the original topic, to seed the next sentence. Using the original topic (rather than
    # the ever-drifting current seed word) keeps it a genuine possibility in every sentence,
    # not just the first one.
    triple = least_likely_trigram(sentence_words, words3_model)
    if triple is None:
        return topic
    choices = [triple[0], triple[1], triple[2], topic]
    return choices[randint(0, 3)]


def main():
    sentences_words = pickle.load(open('sentences_words.pkl', 'rb'))

    forward1_model = build_forward1_model(sentences_words)
    pickle.dump(forward1_model, open('models/forward_1.pkl', 'wb'))

    back1_model = build_backward1_model(sentences_words)
    pickle.dump(back1_model, open('models/backward_1.pkl', 'wb'))

    forward2_model = build_forward2_model(sentences_words)
    pickle.dump(forward2_model, open('models/forward_2.pkl', 'wb'))

    back2_model = build_backward2_model(sentences_words)
    pickle.dump(back2_model, open('models/backward_2.pkl', 'wb'))

    forward3_model = build_forward3_model(sentences_words)
    pickle.dump(forward3_model, open('models/forward_3.pkl', 'wb'))

    back3_model = build_backward3_model(sentences_words)
    pickle.dump(back3_model, open('models/backward_3.pkl', 'wb'))

    words1_model = build_words1_model(sentences_words)
    pickle.dump(words1_model, open('models/words_1.pkl', 'wb'))

    words2_model = build_words2_model(sentences_words)
    pickle.dump(words2_model, open('models/words_2.pkl', 'wb'))

    words3_model = build_words3_model(sentences_words)
    pickle.dump(words3_model, open('models/words_3.pkl', 'wb'))


if __name__ == '__main__':
    main()
