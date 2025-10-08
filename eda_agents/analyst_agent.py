from langchain_google_genai import ChatGoogleGenerativeAI
import os
import pandas as pd
import plotly.express as px
from utils_eda import build_result_block

class AnalystAgentOld:
    def __init__(self, dfs: dict):
        self.dfs = dfs

    def describe(self):
        results = []
        for name, df in self.dfs.items():
            desc = df.describe(include="all").transpose().fillna("NaN")
            results.append({"title": f"📊 Resumo estatístico — {name}", "type": "json", "content": desc.to_dict()})
            results.append({"title": f"ℹ️ Informações — {name}", "type": "json", "content": {"shape": df.shape, "missing": df.isna().
            sum().to_dict(), "dtypes": df.dtypes.astype(str).to_dict()}})
        return results

class AnalystAgentLastWorkingVersion:
    def __init__(self, dfs):
        self.dfs = dfs

    def describe(self):
        results = []
        for name, df in self.dfs.items():
            try:
                # Estatísticas descritivas
                desc = df.describe(include="all").transpose()

                # Reset de índice para mostrar o nome da coluna
                desc = desc.reset_index().rename(columns={"index": "Coluna"})

                results.append({
                    "title": f"📊 Resumo estatístico — {name}",
                    "type": "table",
                    "content": desc
                })

            except Exception as e:
                results.append({
                    "title": f"Erro ao gerar resumo — {name}",
                    "type": "text",
                    "content": str(e)
                })
        return results
    

class AnalystAgentLastWorkingVersion2:

    def __init__(self, dfs):
        self.dfs = dfs

    def describe(self):
        results = []
        for name, df in self.dfs.items():
            try:
                # Estatísticas descritivas
                desc = df.describe(include="all").transpose()
                desc = desc.reset_index().rename(columns={"index": "Coluna"})

                # Bloco 1: tabela resumo
                results.append({
                    "title": f"📊 Resumo estatístico — {name}",
                    "type": "table",
                    "content": desc
                })

                # Bloco 2: alguns gráficos automáticos (se for viável)
                numeric_cols = df.select_dtypes(include="number").columns
                if len(numeric_cols) > 0:
                    fig = px.histogram(df, x=numeric_cols[0], title=f"Distribuição de {numeric_cols[0]}")
                    results.append({
                        "title": f"📈 Distribuição — {numeric_cols[0]}",
                        "type": "chart",
                        "content": fig
                    })

                categorical_cols = df.select_dtypes(include="object").columns
                if len(categorical_cols) > 0:
                    top_col = categorical_cols[0]
                    freq = df[top_col].value_counts().reset_index()
                    freq.columns = [top_col, "Frequência"]
                    fig = px.bar(freq, x=top_col, y="Frequência", title=f"Frequências — {top_col}")
                    results.append({
                        "title": f"📊 Frequências — {top_col}",
                        "type": "chart",
                        "content": fig
                    })

                # Bloco 3: comentário interpretativo
                commentary = []
                commentary.append("📌 Observações iniciais:")
                commentary.append(f"- O dataset **{name}** possui {df.shape[0]} linhas e {df.shape[1]} colunas.")
                if len(numeric_cols) > 0:
                    commentary.append(f"- Há {len(numeric_cols)} variáveis numéricas, exemplo: {', '.join(numeric_cols[:3])}.")
                if len(categorical_cols) > 0:
                    commentary.append(f"- Há {len(categorical_cols)} variáveis categóricas, exemplo: {', '.join(categorical_cols[:3])}.")
                commentary.append("- Use este resumo para verificar qualidade dos dados, outliers e valores ausentes.")

                results.append({
                    "title": f"📝 Comentários sobre {name}",
                    "type": "text",
                    "content": "\n".join(commentary)
                })

            except Exception as e:
                results.append({
                    "title": f"Erro ao gerar resumo — {name}",
                    "type": "text",
                    "content": str(e)
                })

        return results
    



class AnalystAgent:
    def __init__(self, dfs, gemini_api_key):
        self.dfs = dfs
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.2, google_api_key=gemini_api_key)

    def describe(self):
        results = []
        for name, df in self.dfs.items():
            desc = df.describe(include="all").T
            results.append({
                "title": f"📊 Resumo estatístico — {name}",
                "type": "table",
                "content": desc
            })

            prompt = f"Explique resumidamente as estatísticas do dataset {name}: {desc.head().to_dict()}"
            try:
                resp = self.llm.invoke(prompt)
                commentary = resp.content
            except Exception as e:
                commentary = (
                    "Resumo automático indisponível (possível quota excedida). "
                    "Verifique valores ausentes, outliers e distribuições para obter insights iniciais."
                )

            results.append({
                "title": f"📝 Comentários — {name}",
                "type": "text",
                "content": commentary
            })
        return results