# text2emoji

Using a neural net to generate emojis from words. The network takes word embeddings as inputs, and produces pixels as outputs. Using a trained model, you can create a new phrase like "red banana" and see what the emoji would look like.

Since there aren't that many emojis (on the order of a few thousand), there is likely not enough data to train anything useful. Still, though, it's possible to generate hilarious images.

# Results

Here is a table of some cherry-picked generalizations. Keep in mind that the [emoji dataset](https://unicode.org/emoji/charts/full-emoji-list.html) is very specific; there is no "apple" emoji, for example, just "red apple" and "green apple".

| Image | Phrase |
|-------|--------|
| ![apple](samples/apple.png) | apple |
| ![yellow apple](samples/yellow_apple.png) | yellow apple |
| ![red banana](samples/red_banana.png) | red banana |
| ![rainbow cross mark](samples/rainbow_cross_mark.png) | rainbow_cross_mark |
| ![clown face with horns](samples/clown_face_with_horns.png) | clown face with horns |
| ![flower](samples/flower.png) | flower |
| ![rainbow duck](samples/rainbow_duck.png) | rainbow duck |
| ![rainbow bomb](samples/rainbow_bomb.png) | rainbow bomb |
| ![rainbow smiling face with horns](samples/rainbow_smiling_face_with_horns.png) | rainbow smiling face with horns |

Here are some reconstructions from the training set:

![training set reconstructions](samples/training_reconstructions.png)

# Usage

 * Download [glove embeddings](http://nlp.stanford.edu/data/glove.42B.300d.zip) and unzip them into this directory.
 * Run `python fetch.py` to download the emoji data.
 * Run `python run_train.py` to train a model.
 * Run `python run_grid.py` to produce reconstructions from the training set.
 * RUn `python run_eval.py` to produce images for new phrases.
