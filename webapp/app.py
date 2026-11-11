# SPDX-FileCopyrightText: 2026 Mike Coats <mike.coats@nature.scot>
# SPDX-License-Identifier: MIT

from flask import Flask, jsonify, render_template, request

from webapp.generation import autocomplete_suggestions, generate_post, words1

app = Flask(__name__)


@app.route('/')
def index():
    prompt = request.args.get('prompt', '').strip().lower() or None
    post = generate_post(prompt)
    return render_template('index.html', topic=post['topic'], paragraphs=post['paragraphs'])


@app.route('/autocomplete')
def autocomplete():
    partial_word = request.args.get('q', '').strip().lower()
    if not partial_word:
        return jsonify([])
    words = autocomplete_suggestions(partial_word)
    # Pair each suggestion with its occurrence count, so the dropdown can show the word's
    # frequency in the source material alongside it.
    return jsonify([{'word': word, 'count': words1[word]} for word in words])


if __name__ == '__main__':
    app.run(debug=True)
