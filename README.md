<!--
SPDX-FileCopyrightText: 2026 Mike Coats <mike.coats@nature.scot>
SPDX-License-Identifier: MIT
-->

# Very Small Language Model

The Robots Are (Maybe Not) Taking Over

- Building a
  ~~Large~~ Very Small Language Model
- with
  ~~Artificial Intelligence~~ Algorithmic Inertia

## Instructions

```shell
$ python -m venv .venv --prompt=vslm

$ source .venv/Scripts/activate

$ pip install -e .[optional]

$ jupyter lab
```

### Running the demo web app

Once the models in `models/` have been built (by running `python vslm/language_model.py`, which
itself needs `sentences_words.pkl` from `02_training_data.ipynb`), a small Flask app is available
to present a generated post as a webpage:

```shell
$ flask --app webapp.app run
```

Then visit <http://127.0.0.1:5000/> in a browser. Each page load generates a new post; pass a
`prompt` query parameter (e.g. `http://127.0.0.1:5000/?prompt=peatland`) to seed it with a
specific topic word instead of a random one.

## License

Unless stated otherwise, the codebase is released under the [MIT License](LICENSES/MIT.txt).
