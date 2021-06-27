# Preprocessing

Review data was preprocessed in the following manner.

## Basic text cleaning
The basic cleaning of reviews included:
- Removal of urls: almost never contribute to sentiment of review.
- Removal of mail adresses: unnecessary 
- Swap new line characters for spaces.
- Remove single quotes.

## Normalization
- Lowercase all text
- Remove stopwords
- Stemming

## Tokenization
- Splitting review up in sentences
