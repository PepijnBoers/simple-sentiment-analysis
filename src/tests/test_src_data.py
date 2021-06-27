import pytest
import pandas as pd
from src.data.convert_data import parse_data


@pytest.fixture
def raw_data_sample():
    sample_df = pd.read_csv(
        "../data/raw/train_sample.txt", delimiter="\n", names=["unfiltered_review"]
    )
    return sample_df


def test_converter(raw_data_sample):
    converted_df = raw_data_sample.apply(parse_data, axis=1)

    assert converted_df["label"][0] == "__label__2"
    assert converted_df["title"][0] == "Stuning even for the non-gamer"
    assert (
        converted_df["body"][0]
        == " This sound track was beautiful! It paints the senery in your mind so well I would recomend it even to people who hate vid. game music! I have played the game Chrono Cross but out of all of the games I have ever played it has the best music! It backs away from crude keyboarding and takes a fresher step with grate guitars and soulful orchestras. It would impress anyone who cares to listen! ^_^"
    )
