from pathlib import Path

import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

from utils import compute_embeddings, load_data, load_embeddings, load_model

st.set_page_config(
    layout="wide",
    page_title="Mission Transition - Test du document embedding pour la recherche",
    page_icon="🔍",
)
DATA_PATH = Path("data/")
MT_AIDES_FILEPATH = DATA_PATH / "MT_aides.json"
EMBEDDINGS_FILEPATH = DATA_PATH / "aides_embeddings.pkl"

aides_df = load_data(MT_AIDES_FILEPATH)
aides_text = aides_df["full_text"].tolist()

model = load_model("dangvantuan/sentence-camembert-base")
aides_embeddings = load_embeddings(EMBEDDINGS_FILEPATH)

query = st.text_input("Recherche :")


cols_up, cols_down = st.columns(3), st.columns(3)

if query:
    input_embedding = model.encode(query)
    similarities = cosine_similarity([input_embedding], aides_embeddings)
    aides_df["similarity"] = similarities[0]
    top_6_df = aides_df.sort_values("similarity", ascending=False).head(6)

    for i, (_, item) in enumerate(top_6_df.iterrows()):
        if i < 3:
            cols_up[i].subheader(item["name"])
            cols_up[i].markdown(f"Similarity: **{item['similarity']}**")
            cols_up[i].expander("Voir la description").markdown(
                item["description"], unsafe_allow_html=True
            )
        else:
            cols_down[i - 3].subheader(item["name"])
            cols_down[i - 3].markdown(f"Similarity: **{item['similarity']}**")
            cols_down[i - 3].expander("Voir la description").markdown(
                item["description"], unsafe_allow_html=True
            )
else:
    st.info("Entrez un terme de recherche pour commencer.")
