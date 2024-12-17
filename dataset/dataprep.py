import google.generativeai as genai
import matplotlib.pyplot as plt
import requests
import json
from tqdm import tqdm
import os
import pandas as pd
from dotenv import load_dotenv
load_dotenv('../.env')
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import google.generativeai as genai
url_base = 'https://dadosabertos.camara.leg.br/api/v2/'
genai.configure(api_key=os.environ['GEMINI_KEY'])
model = genai.GenerativeModel("gemini-1.5-flash")


# Exercicio 1 - Arquitetura

"""
A arquitetura da solução inclui fontes de dados como Parquet, JSON e FAISS, que alimentam pontos de processamento onde os dados são pré-processados, analisados e sumarizados usando o LLM "gemini-1.5-flash". O modelo gera resumos e insights, que são configurados via arquivos YAML. O Dashboard apresenta esses dados nas abas Overview, Despesas e Proposições, com gráficos e relatórios interativos. A base de dados FAISS armazena embeddings para consultas rápidas. LLMs são usados principalmente para sumarização de textos e extração de insights a partir de grandes volumes de dados.
"""

# Exercicio 2 - Resposta do modelo salvo no arquivo img

"""
As respostas dos três modelos tem o mesmo foco, que é explicar o papel da Câmara dos Deputados, mas cada um tem uma abordagem diferente. O Claude-3.5 foca em aspectos mais formais e judiciais, como a autorização de processos contra o presidente e a iniciativa popular, dando um tom mais direto e objetivo. Já o Gemini-1.5 é mais abrangente e foca no impacto que a Câmara tem na vida dos cidadãos, tocando em áreas como economia, saúde e educação, de forma mais acessível e menos técnica. O GPT-4o combina aspectos técnicos e claros, detalhando por exemplo a quantidade mínima e maxima de deputados por estado e explicando o sistema proporcional de eleição. Ele é mais estruturado e organizado, trazendo um equilíbrio entre dados específicos e uma visão mais ampla do papel da câmara no sistema democrático. A principal diferença está no estilo: o Claude é mais direto, o Gemini mais discursivo e o GPT-4o e mais técnico e informativo.

"""

# Exercicio 3
def salvar_deputados_em_parquet():
    url = url_base + 'deputados'
    response = requests.get(url)

    if response.status_code == 200:
        df_deputados = pd.DataFrame.from_dict(json.loads(response.text)['dados'])
        df_deputados.to_parquet('../data/deputados.parquet', index=False)
        print("Arquivo 'deputados.parquet' salvo com sucesso!")
    else:
        print(f"Erro ao acessar a API: {response.status_code}")

salvar_deputados_em_parquet()

def grafico_pizza_deputados():
    prompt_start = """
    Os '../data/deputados.parquet' é um arquivo que contém uma coluna chamada 'siglaPartido', que armazena os partidos dos deputados. Quero um código Python **válido e sem explicações extras** que plote um gráfico de pizza com a distribuição dos deputados por partido. E salva em '../docs/distribuicao_deputados.png
    """

    response = model.generate_content(prompt_start)
    analysis_code = response.text.replace("```python\n", "").replace("\n```", "")
    exec(analysis_code)

grafico_pizza_deputados()

def dist_deputados():
    prompt_start = """
    Os no arquivo "../data/deputados.parquet" contém uma coluna chamada 'siglaPartido', que armazena os partidos dos deputados. Quero um codigo python **válido e sem explicações extras** da distribuição dos deputados por partido. Gere insights sobre a distribuição de partidos e como isso influencia a câmara. 
    Salve-a em '../data/insights_distribuicao_deputados.json'.
    """

    response = model.generate_content(prompt_start)
    analysis_code = response.text.replace("```python\n", "").replace("\n```", "")
    exec(analysis_code)

dist_deputados()

# Exercicio 4

def dados_depesas():
    url = url_base + 'deputados'
    response = requests.get(url)
    df_deputados = pd.DataFrame.from_dict(json.loads(response.text)['dados'])
    list_expenses = []
    anoDespesa = '2024'
    maxItens = '100'
    mesDespesa = '8'
    for id in df_deputados.id.unique():
        url = f'{url_base}/deputados/{id}/despesas'
        params = {
            'ano': anoDespesa,
            'itens': maxItens,
            'mes': mesDespesa,
        }
        response = requests.get(url)
        df_resp = pd.DataFrame().from_dict(json.loads(response.text)['dados'])
        df_resp['id'] = id
        list_expenses.append(df_resp)
        df_links = pd.DataFrame().from_dict(json.loads(response.text)['links'])
        df_links = df_links.set_index('rel').href
        while 'next' in df_links.index:
            response = requests.get(df_links['next'])
            df_resp = pd.DataFrame().from_dict(json.loads(response.text)['dados'])
            df_resp['id'] = id
            list_expenses.append(df_resp)
            df_links = pd.DataFrame().from_dict(json.loads(response.text)['links'])
            df_links = df_links.set_index('rel').href
    df_expenses = pd.concat(list_expenses)
    df_expenses = df_expenses.merge(df_deputados, on=['id'])
    df_expenses.to_parquet('../data/serie_despesas_diarias_deputados.parquet', index=False)
dados_depesas()

def llm_despesas():
    prompt_start = """
    Você é um cientista de dados para gerar insights de acordo com a estatística e análises realizadas na base de dados.
    Eu tenho um conjunto de dados de despesas de deputados, armazenados no arquivo '../data/serie_despesas_diarias_deputados.parquet'. O arquivo contém as seguintes colunas:

    1. 'ano' - Ano da despesa.
    2. 'mes' - Mês da despesa.
    3. 'tipoDespesa' - Tipo de despesa realizada.
    4. 'codDocumento' - Código do documento da despesa.
    5. 'tipoDocumento' - Tipo do documento da despesa.
    6. 'codTipoDocumento' - Código do tipo de documento.
    7. 'dataDocumento' - Data em que a despesa foi realizada.
    8. 'numDocumento' - Número do documento da despesa.
    9. 'valorDocumento' - Valor total da despesa.
    10. 'urlDocumento' - URL do documento da despesa.
    11. 'nomeFornecedor' - Nome do fornecedor da despesa.
    12. 'cnpjCpfFornecedor' - CNPJ ou CPF do fornecedor.
    13. 'valorLiquido' - Valor líquido da despesa.
    14. 'valorGlosa' - Valor glosado da despesa.
    15. 'numRessarcimento' - Número do ressarcimento da despesa.
    16. 'codLote' - Código do lote da despesa.
    17. 'parcela' - Número da parcela da despesa.
    18. 'id' - ID da despesa.
    19. 'uri' - URI da despesa.
    20. 'nome' - Nome do deputado responsável pela despesa.
    21. 'siglaPartido' - Sigla do partido do deputado.
    22. 'uriPartido' - URI do partido do deputado.
    23. 'siglaUf' - Sigla da unidade federativa (UF) do deputado.
    24. 'idLegislatura' - ID da legislatura.
    25. 'urlFoto' - URL da foto do deputado.
    26. 'email' - E-mail do deputado.

    Gostaria que você realizasse até 3 análises nos dados de despesas

    Forneça o código Python necessário para realizar essas análises e utilize subplots.
    Forneça apenas o código Python
    """
    response = model.generate_content(prompt_start)
    analysis_code = response.text.replace("```python\n", "").replace("\n```", "")
    exec(analysis_code)

llm_despesas()

def llm_insights():
    prompt_start = """
    Você é um cientista de dados para gerar insights de acordo com a estatística e análises realizadas na base de dados.
    Eu tenho um conjunto de dados de despesas de deputados, armazenados no arquivo '../data/serie_despesas_diarias_deputados.parquet'. O arquivo contém as seguintes colunas:

    1. 'ano' - Ano da despesa.
    2. 'mes' - Mês da despesa.
    3. 'tipoDespesa' - Tipo de despesa realizada.
    4. 'codDocumento' - Código do documento da despesa.
    5. 'tipoDocumento' - Tipo do documento da despesa.
    6. 'codTipoDocumento' - Código do tipo de documento.
    7. 'dataDocumento' - Data em que a despesa foi realizada.
    8. 'numDocumento' - Número do documento da despesa.
    9. 'valorDocumento' - Valor total da despesa.
    10. 'urlDocumento' - URL do documento da despesa.
    11. 'nomeFornecedor' - Nome do fornecedor da despesa.
    12. 'cnpjCpfFornecedor' - CNPJ ou CPF do fornecedor.
    13. 'valorLiquido' - Valor líquido da despesa.
    14. 'valorGlosa' - Valor glosado da despesa.
    15. 'numRessarcimento' - Número do ressarcimento da despesa.
    16. 'codLote' - Código do lote da despesa.
    17. 'parcela' - Número da parcela da despesa.
    18. 'id' - ID da despesa.
    19. 'uri' - URI da despesa.
    20. 'nome' - Nome do deputado responsável pela despesa.
    21. 'siglaPartido' - Sigla do partido do deputado.
    22. 'uriPartido' - URI do partido do deputado.
    23. 'siglaUf' - Sigla da unidade federativa (UF) do deputado.
    24. 'idLegislatura' - ID da legislatura.
    25. 'urlFoto' - URL da foto do deputado.
    26. 'email' - E-mail do deputado.

    Gere uma lista de 3 análises que podem ser implementadas de acordo com os serie_despesas_diarias_deputados.parquet disponíveis, Salve o resultado como um JSON (../data/insights_despesas_deputados.json):
    {[
        {'Nome':'nome da análise',
        'Objetivo': 'o que precisamos analisar',
        'Método': 'como o analisamos',
        'Resultado': 'resultado da análise'
        }
    ]
    }
    Faça sem explicação desnecessaria, apenas o código.
    """
    response = model.generate_content(prompt_start)
    analysis_code = response.text.replace("```python\n", "").replace("\n```", "")
    exec(analysis_code)
llm_insights()

# Exercicio 5

def coletar_proposicoes(data_inicio, data_fim, tema):
    url = f'https://dadosabertos.camara.leg.br/api/v2/proposicoes?dataInicio={data_inicio}&dataFim={data_fim}&codTema={tema}&ordem=ASC&ordenarPor=id'
    proposicoes = []
    response = requests.get(url)
    if response.status_code == 200:
        dados = response.json()
        proposicoes = dados['dados'][:10]
    else:
        print(f"Erro na requisição para o tema {tema}: {response.status_code}")
    return proposicoes

def salvar_proposicoes_em_parquet(proposicoes, file_path):
    df = pd.DataFrame(proposicoes)
    df.to_parquet(file_path, index=False)

data_inicio = "2024-08-01"
data_fim = "2024-08-31"
temas = [40, 46, 62]
todas_proposicoes = []

for tema in temas:
    proposicoes = coletar_proposicoes(data_inicio, data_fim, tema)
    todas_proposicoes.extend(proposicoes)
salvar_proposicoes_em_parquet(todas_proposicoes, "../data/proposicoes_deputados.parquet")

def summary_chunk_parquet():
    arquivo = "../data/proposicoes_deputados.parquet"
    resultado = "../data/sumarizacao_proposicoes.json"
    df = pd.read_parquet(arquivo)
    required_columns = {'siglaTipo', 'numero', 'ano', 'ementa'}
    if not required_columns.issubset(df.columns):
        print("O arquivo não contém todas as colunas necessárias: siglaTipo, numero, ano, ementa")
        return
    summaries = df.apply(
        lambda row: f"{row['siglaTipo']} {row['numero']}/{row['ano']}: {row['ementa']}", axis=1
    )
    summaries_list = summaries.tolist()

    with open(resultado, "w", encoding="utf-8") as json_file:
        json.dump(summaries_list, json_file, ensure_ascii=False, indent=4)
    print(f"Arquivo salvo com sucesso em: {resultado}")
summary_chunk_parquet()