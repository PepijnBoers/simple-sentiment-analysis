import pytest
import pandas as pd
from src.features.preprocess import preprocess
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


@pytest.fixture
def raw_data_sample():
    sample_df = pd.read_csv(
        "../data/interim/train_sample.csv",
    )
    return sample_df


def test_preprocessing(raw_data_sample):
    # Load stopwords and stemmer.
    stop_words = set(stopwords.words("english"))
    ps = PorterStemmer()

    # Preprocess
    preprocessed_df = raw_data_sample.progress_apply(
        preprocess, args=(ps, stop_words), axis=1
    )

    assert (
        preprocessed_df["review"][0]
        == "stune even non-gam sound track beauti ! paint seneri mind well would recomend even peopl hate vid . game music ! play game chrono cross game ever play best music ! back away crude keyboard take fresher step grate guitar soul orchestra . would impress anyon care listen ! ^_^"
    )
    assert (
        preprocessed_df["review"][1]
        == "best soundtrack ever anyth . im read lot review say best game soundtrack figur id write review disagre bit . opinino yasunori mitsuda ultim masterpiec . music timeless im listen year beauti simpli refus fade.th price tag pretti stagger must say , go buy cd much money , one feel would worth everi penni ."
    )
