# Amazon Reviews for Sentiment Analysis

Kaggle dataset containing 4 million Amazon customer reviews that are divided into `__label__1` (1-2 stars) and `__label__2` (4-5 stars). The data can be downloaded from the [Kaggle website](https://www.kaggle.com/bittlingmayer/amazonreviews) and requires you to login.


1. Extract data
```
bunzip2 train.ft.txt.bz2
bunzip2 test.ft.txt.bz2
```

2. Create quick train/test set, 250/50 reviews each.
```bash
head -n250 train.ft.txt > train250.txt
head -n50 test.ft.txt > test50.txt
```

3. Parse data
```bash
python convert_data.py -i /Users/pboers/projects/sentiment-analysis-dl/data/raw/train250.txt -o /Users/pboers/projects/sentiment-analysis-dl/data/interim/train250
python convert_data.py -i /Users/pboers/projects/sentiment-analysis-dl/data/raw/test50.txt -o /Users/pboers/projects/sentiment-analysis-dl/data/interim/test50
```

4. Preprocess data.
```bash
python preprocess.py -i /Users/pboers/projects/sentiment-analysis-dl/data/interim/train250.csv -o /Users/pboers/projects/sentiment-analysis-dl/data/processed/train250
python preprocess.py -i /Users/pboers/projects/sentiment-analysis-dl/data/interim/test50.csv -o /Users/pboers/projects/sentiment-analysis-dl/data/processed/test50
```
