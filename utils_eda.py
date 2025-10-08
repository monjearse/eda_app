import os, zipfile
import pandas as pd

def read_any(files):
    dfs = {}
    def add_df(path, df):
        name = os.path.splitext(os.path.basename(path))[0]
        df.columns = df.columns.str.strip()
        dfs[name] = df

    for f in files:
        fname = getattr(f, "name", str(f))
        if str(fname).lower().endswith(".zip"):
            with zipfile.ZipFile(f) as z:
                for member in z.namelist():
                    if member.lower().endswith(".csv"):
                        with z.open(member) as h:
                            df = pd.read_csv(h, sep=None, engine="python")
                            add_df(member, df)
        elif str(fname).lower().endswith(".csv"):
            df = pd.read_csv(f, sep=None, engine="python")
            add_df(fname, df)
    return dfs


def build_result_block(block_type: str, title: str, content):
    """
    Cria um bloco padronizado para retorno dos agentes.

    :param block_type: "table", "chart", "text", "json"
    :param title: título a ser exibido no Streamlit
    :param content: conteúdo (DataFrame, figura Plotly, string, dict, etc.)
    :return: dict padronizado
    """
    return {
        "title": title,
        "type": block_type,
        "content": content
    }
