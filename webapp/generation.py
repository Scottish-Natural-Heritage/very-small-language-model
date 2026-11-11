# SPDX-FileCopyrightText: 2026 Mike Coats <mike.coats@nature.scot>
# SPDX-License-Identifier: MIT

"""Generate NatureScot-style blog posts from the trained n-gram models.

This mirrors the generation logic built up in ``05_prompting_our_model.ipynb``
(unigram/bigram/trigram sentence generation, chained into a paragraph via the
least-likely-but-pre-occurring trigram). The actual model-building/generation
primitives live in ``vslm.language_model`` and are reused here (instead of
being duplicated) so the Flask app can build a whole multi-paragraph "post".
"""

import pickle
from pathlib import Path
from random import choice, randint

from vslm import language_model
from vslm.language_model import MAX_SENTENCES_PER_PARAGRAPH, MIN_SENTENCES_PER_PARAGRAPH

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / 'models'

# A "post" is made up of several paragraphs.
MIN_PARAGRAPHS_PER_POST = 5
MAX_PARAGRAPHS_PER_POST = 9


def _load_model(filename):
    with (MODELS_DIR / filename).open('rb') as model_file:
        return pickle.load(model_file)


forward1 = _load_model('forward_1.pkl')
backward1 = _load_model('backward_1.pkl')
forward2 = _load_model('forward_2.pkl')
backward2 = _load_model('backward_2.pkl')
forward3 = _load_model('forward_3.pkl')
backward3 = _load_model('backward_3.pkl')
words1 = _load_model('words_1.pkl')
words3 = _load_model('words_3.pkl')
vocabulary = [word for word in words1 if word not in ('START', 'END')]

# Only offer words that occur at least this many times in the sources as a random seed word,
# so a brand new post doesn't start from a rare, one-off token.
MIN_SEED_WORD_OCCURRENCES = 5
common_vocabulary = [word for word in vocabulary if words1[word] >= MIN_SEED_WORD_OCCURRENCES]

# How many words to offer in the topic autocomplete dropdown.
AUTOCOMPLETE_LIMIT = 10

# Only suggest words that occur at least this many times in the sources, so rare, one-off
# tokens don't clutter the autocomplete dropdown.
MIN_AUTOCOMPLETE_OCCURRENCES = 3


def generate_trigram_sentence(seed_word):
    # Build one full sentence (as a list of words) using the trigram models.
    return language_model.generate_trigram_sentence(
        seed_word, forward3, forward2, forward1, backward3, backward2, backward1
    )


def next_seed_word(sentence_words, topic):
    # Pick one of the three words from the least likely (but still pre-occurring) trigram,
    # or the original topic, to seed the next sentence. Using the original topic (rather than
    # the ever-drifting current seed word) keeps it a genuine possibility in every sentence,
    # not just the first one.
    return language_model.next_seed_word(sentence_words, topic, words3)


def random_seed_word():
    return choice(common_vocabulary)


def autocomplete_suggestions(partial_word, limit=AUTOCOMPLETE_LIMIT):
    # Find vocabulary words starting with the given (lowercased) partial word and occurring at
    # least MIN_AUTOCOMPLETE_OCCURRENCES times in the corpus, ordered by how often they occur
    # (most common first), for use as a topic autocomplete feed.
    partial_word = partial_word.lower()
    matches = [
        word
        for word in vocabulary
        if word.startswith(partial_word) and words1[word] >= MIN_AUTOCOMPLETE_OCCURRENCES
    ]
    matches.sort(key=lambda word: words1[word], reverse=True)
    return matches[:limit]


def generate_paragraph(seed_word, topic):
    # Generate a paragraph: each sentence is built with the trigram method, then the least
    # statistically likely (but still pre-occurring) trigram in it is found and one of its
    # three words (or the original topic) seeds the next sentence. The number of sentences is
    # sampled from the sensible sentences-per-paragraph range observed in our sources.
    sentences = []
    for _ in range(randint(MIN_SENTENCES_PER_PARAGRAPH, MAX_SENTENCES_PER_PARAGRAPH)):
        sentence_words = generate_trigram_sentence(seed_word)
        sentence_text = ' '.join(sentence_words) + '.'
        sentences.append(sentence_text.capitalize())
        seed_word = next_seed_word(sentence_words, topic)

    return ' '.join(sentences), seed_word


def generate_post(prompt=None):
    # Generate a whole "post": a topic word and several generated paragraphs, each one seeded
    # by a word carried over from the previous paragraph's final sentence, with the original
    # topic kept as a perpetual possibility in every sentence (not just the first).
    seed_word = prompt or random_seed_word()
    topic = seed_word
    paragraphs = []
    for _ in range(randint(MIN_PARAGRAPHS_PER_POST, MAX_PARAGRAPHS_PER_POST)):
        paragraph_text, seed_word = generate_paragraph(seed_word, topic)
        paragraphs.append(paragraph_text)

    return {'topic': topic, 'paragraphs': paragraphs}
