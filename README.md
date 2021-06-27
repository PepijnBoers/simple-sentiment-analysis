Exploration of different models for sentiment analysis using Amazon reviews
==============================
Simple comparison of different sentimement analysis models to get familiar with different techniques. The comparison contains:
- Logistic Regression
- Long Short-Term Memory network (type of RNN)
- Convolutional neural network
- BERT for sentiment analysis

## Data
A minor subset of Kaggle's Amazon customer reviews dataset, more about the data [here](docs/data.md).

Example of 1-2 star review:
```
This is a self-published book, and if you want to know why--read a few paragraphs! Those 5 star reviews must have been written by Ms. Haddon's family and friends--or perhaps, by herself! I can't imagine anyone reading the whole thing--I spent an evening with the book and a friend and we were in hysterics reading bits and pieces of it to one another. It is most definitely bad enough to be entered into some kind of a "worst book" contest. I can't believe Amazon even sells this kind of thing. Maybe I can offer them my 8th grade term paper on "To Kill a Mockingbird"--a book I am quite sure Ms. Haddon never heard of. Anyway, unless you are in a mood to send a book to someone as a joke---stay far, far away from this one!
```

Example of 4-5 star review:
```
This sound track was beautiful! It paints the senery in your mind so well I would recomend it even to people who hate vid. game music! I have played the game Chrono Cross but out of all of the games I have ever played it has the best music! It backs away from crude keyboarding and takes a fresher step with grate guitars and soulful orchestras. It would impress anyone who cares to listen! ^_^
```

## Training
Training (fitting) scripts were written for each model and are placed in `src/models/train_model_*.py`. The **LSTM** and **CNN** model were both trained in a similar manner using Keras and could be selected by passing the `--model` parameter when running the script. After training had finished, models were stored in the `/models` section.

For example when running the train script for the LSTM model use:
```bash
python train_model_keras.py -i <path-to-data/processed/data.csv> --max-words 5000 --max-length 200 --epochs 20 --model lstm
```


## Results
The trained models could be used for prediction using the `src/models/predict_model_*.py` scripts. Each model was tested on a subset of the test data and was judged in terms of precision, recall and f1-score using a confusion matrix. See the results below:

| Model               | Precision | Recall  | F1-score | Accuracy |
|---------------------|-----------|---------|----------|----------|
| Logistic Regression | 0.7778    | 0.83333 | 0.8046   | 0.7733   |
| LSTM                | 0.6970    | 0.8214  | 0.7541   | 0.7000   |
| CNN                 | 0.6667    | **0.9286**  | 0.7761   | 0.7000   |
| BERT                | **0.9600**    | 0.8571  | **0.9057**   | **0.9000**   |

### Confusion matrices
<p float="left">
  <img src="reports/figures/Confusion Matrix Logistic Regression.png" width="45%" />
  <img src="reports/figures/Confusion Matrix LSTM.png" width="45%" /> 
</p>

<p float="left">
  <img src="reports/figures/Confusion Matrix CNN.png" width="45%" />
  <img src="reports/figures/Confusion Matrix BERT.png" width="45%" /> 
</p>


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
