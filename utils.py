from os import PathLike
from typing import List, Union

import numpy as np
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer


def remove_html_tags(text: str) -> str:
    """
    Remove HTML tags using beautifulsoup.
    """
    soup = BeautifulSoup(text, "lxml")
    return soup.text


def normalize_text(text: str) -> str:
    """
    Remove newlines
    """
    text = remove_html_tags(text)
    return text.replace("\n", "").strip()


@st.cache(allow_output_mutation=True)
def load_data(data_path: Union[str, PathLike]) -> pd.DataFrame:

    aides_df = pd.read_json(data_path)
    aides_df["full_text"] = (
        aides_df["name"] + " " + aides_df["description"].apply(normalize_text)
    )
    return aides_df


@st.cache(allow_output_mutation=True)
def load_model(
    model_name: str = "dangvantuan/sentence-camembert-large",
) -> SentenceTransformer:
    model = SentenceTransformer(model_name)
    model.max_seq_length = 512
    return model


@st.cache
def compute_embeddings(texts_list: List[str], model: SentenceTransformer) -> np.ndarray:
    aides_vectors = model.encode(texts_list, batch_size=16)
    return aides_vectors
