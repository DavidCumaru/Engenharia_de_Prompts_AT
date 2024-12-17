import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import json
import pyarrow.parquet as pq
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

def load_data():
    with open('../data/insights_despesas_deputados.json', 'r', encoding="ISO-8859-1") as f:
        insights_despesas = json.load(f)
    despesas_df = pq.read_table('../data/serie_despesas_diarias_deputados.parquet').to_pandas()
    despesas_df['dataDocumento'] = pd.to_datetime(despesas_df['dataDocumento'])
    proposicoes_df = pq.read_table('../data/proposicoes_deputados.parquet').to_pandas()
    with open('../data/sumarizacao_proposicoes.json', 'r', encoding="ISO-8859-1") as f:
        sumarizacao_proposicoes = json.load(f)

    return insights_despesas, despesas_df, proposicoes_df, sumarizacao_proposicoes

@st.cache_resource
def setup_embedding_model_and_index():
    model_name = 'neuralmind/bert-base-portuguese-cased'
    embedding_model = SentenceTransformer(model_name, device='cpu')

    with open('../data/insights_despesas_deputados.json', 'r', encoding="ISO-8859-1") as f:
        insights_despesas = json.load(f)

    despesas_df = pq.read_table('../data/serie_despesas_diarias_deputados.parquet').to_pandas()
    deputados_df = pq.read_table('../data/deputados.parquet').to_pandas()
    proposicoes_df = pq.read_table('../data/proposicoes_deputados.parquet').to_pandas()

    with open('../data/sumarizacao_proposicoes.json', 'r', encoding="ISO-8859-1") as f:
        sumarizacao_proposicoes = json.load(f)

    texts = [
        json.dumps(insights_despesas),
        deputados_df.to_string(),
        despesas_df.to_string(),
        proposicoes_df.to_string(),
        json.dumps(sumarizacao_proposicoes)
    ]
    embeddings = embedding_model.encode(texts)
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)

    return embedding_model, index, texts

def chat_with_assistant(embedding_model, index, texts, user_query):
    def generate_self_ask_query(query):
        return f"{query} "
    self_ask_query = generate_self_ask_query(user_query)
    query_embedding = embedding_model.encode(self_ask_query).reshape(1, -1)
    D, I = index.search(query_embedding, k=1)
    closest_text = texts[I[0][0]]
    return closest_text

def despesas_page(insights_despesas, despesas_df):
    st.title("Despesas dos Deputados")
    st.write(insights_despesas)
    deputados = despesas_df['nome'].unique()
    selected_deputado = st.selectbox("Selecione um Deputado:", deputados)
    filtered_df = despesas_df[(despesas_df['nome'] == selected_deputado) &
                              (despesas_df['dataDocumento'].dt.month == 8) &
                              (despesas_df['dataDocumento'].dt.year == 2024)]
    if not filtered_df.empty:
        plt.figure(figsize=(10, 6))
        plt.bar(filtered_df['dataDocumento'], filtered_df['valorDocumento'])
        plt.xlabel("Data")
        plt.ylabel("Valor da Despesa")
        plt.title(f"Despesas de {selected_deputado} em Agosto de 2024")
        plt.xticks(rotation=45)
        st.pyplot(plt)
    else:
        st.info(f"Não há dados de despesas para {selected_deputado} em Agosto de 2024.")


def proposicoes_page(proposicoes_df, sumarizacao_proposicoes):
    st.title("Proposições dos Deputados")
    st.write("Dados das Proposições:")
    st.dataframe(proposicoes_df)
    st.write("Resumo das Proposições:")
    st.write(sumarizacao_proposicoes)
    st.subheader("Assistente Virtual Especialista em Câmara dos Deputados")
    user_query = st.text_input("Digite sua pergunta:", placeholder="Ex: Quais são as proposições sobre tecnologia?")
    if st.button("Enviar"):
        if user_query:
            with st.spinner("Consultando o assistente..."):
                embedding_model, index, texts = setup_embedding_model_and_index()
                response = chat_with_assistant(embedding_model, index, texts, user_query)
                st.write("**Resposta do Assistente:**")
                st.text(response)
        else:
            st.warning("Por favor, digite uma pergunta antes de enviar.")

def main():
    insights_despesas, despesas_df, proposicoes_df, sumarizacao_proposicoes = load_data()
    if insights_despesas is None:
        return
    page = st.sidebar.radio("Selecione uma aba:", ("Despesas", "Proposições"))
    if page == "Despesas":
        despesas_page(insights_despesas, despesas_df)
    elif page == "Proposições":
        proposicoes_page(proposicoes_df, sumarizacao_proposicoes)

if __name__ == "__main__":
    main()