// SPDX-FileCopyrightText: 2026 Mike Coats <mike.coats@nature.scot>
// SPDX-License-Identifier: MIT

// Makes the post's topic word (the italicised word in the header) clickable, turning it into a
// text input with an autocomplete suggestion list fed by the vocabulary in models/words_1.pkl
// (via the /autocomplete endpoint), ordered by how often each word occurs - most common first.
document.addEventListener('DOMContentLoaded', function () {
    var display = document.getElementById('topic-display');
    var editor = document.getElementById('topic-editor');
    var input = document.getElementById('topic-input');
    var suggestions = document.getElementById('topic-suggestions');

    if (!display || !editor || !input || !suggestions) {
        return;
    }

    function clearSuggestions() {
        suggestions.innerHTML = '';
    }

    function navigateToTopic(word) {
        window.location.href = '/?prompt=' + encodeURIComponent(word);
    }

    function showEditor() {
        display.hidden = true;
        editor.hidden = false;
        input.focus();
        input.select();
    }

    function hideEditor() {
        editor.hidden = true;
        display.hidden = false;
        clearSuggestions();
    }

    function renderSuggestions(wordCounts) {
        clearSuggestions();
        wordCounts.forEach(function (wordCount) {
            var word = wordCount.word;
            var item = document.createElement('li');
            // Show how many times the word occurs in the source material, e.g. "peatland (42)".
            item.textContent = word + ' (' + wordCount.count + ')';
            // mousedown (rather than click) fires before the input's blur event, so the
            // suggestion is still read before the editor gets hidden again.
            item.addEventListener('mousedown', function (event) {
                event.preventDefault();
                navigateToTopic(word);
            });
            suggestions.appendChild(item);
        });
    }

    function fetchSuggestions(partialWord) {
        if (!partialWord) {
            clearSuggestions();
            return;
        }
        fetch('/autocomplete?q=' + encodeURIComponent(partialWord))
            .then(function (response) {
                return response.json();
            })
            .then(renderSuggestions)
            .catch(clearSuggestions);
    }

    display.addEventListener('click', showEditor);
    display.addEventListener('keydown', function (event) {
        if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault();
            showEditor();
        }
    });

    input.addEventListener('input', function () {
        fetchSuggestions(input.value.trim());
    });

    input.addEventListener('keydown', function (event) {
        if (event.key === 'Enter') {
            event.preventDefault();
            var word = input.value.trim();
            if (word) {
                navigateToTopic(word);
            }
        } else if (event.key === 'Escape') {
            hideEditor();
        }
    });

    input.addEventListener('blur', function () {
        hideEditor();
    });
});
