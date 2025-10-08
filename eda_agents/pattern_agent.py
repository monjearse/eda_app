import os
import plotly.express as px
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI


class PatternAgent:
    """
    Agente responsável por identificar padrões e relações nos dados:
    - Correlações numéricas (fortes, fracas, positivas e negativas)
    - Padrões de frequência em variáveis categóricas
    Sempre prioriza o uso do LLM (Gemini) para interpretação dos padrões,
    com fallback técnico local e visualizações de apoio.
    """

    def __init__(self, dfs, gemini_api_key):
        self.dfs = dfs
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")
        self.llm = ChatGoogleGenerativeAI(
            model=model_name, temperature=0.2, google_api_key=gemini_api_key
        )

    # ============================================================
    # 🔹 MÉTODO PRINCIPAL — CORRELAÇÕES
    # ============================================================
    def correlations(self):
        results = []
        for name, df in self.dfs.items():
            try:
                # Matriz de correlação
                corr = df.corr(numeric_only=True)

                # 🔸 LLM PRIORITÁRIO
                prompt = (
                    f"Você é um analista de dados. Analise a matriz de correlação do dataset '{name}'. "
                    f"Descreva, de forma clara e objetiva, as relações mais fortes (positivas e negativas), "
                    f"indicando possíveis implicações. Use uma linguagem humana e precisa. "
                    f"Matriz de correlação: {corr.to_dict()}."
                )
                commentary = self.llm.invoke(prompt).content

                # 🔹 Gráfico de correlação (heatmap)
                fig = px.imshow(
                    corr,
                    text_auto=True,
                    color_continuous_scale="RdBu_r",
                    title=f"Matriz de Correlação — {name}",
                )

                results.append({
                    "title": f"🔗 Matriz de Correlação — {name}",
                    "type": "chart",
                    "content": fig
                })
                results.append({
                    "title": f"🧠 Interpretação Automática — {name}",
                    "type": "text",
                    "content": commentary
                })

            except Exception:
                # 🔸 FALLBACK LOCAL
                corr = df.corr(numeric_only=True)
                top_corr = (
                    corr.where(~corr.isna())
                    .unstack()
                    .dropna()
                    .sort_values(ascending=False)
                )

                top_pairs = [
                    f"{a}–{b}: {v:.2f}" for (a, b), v in top_corr.head(5).items() if a != b
                ]

                fig = px.imshow(
                    corr,
                    text_auto=True,
                    color_continuous_scale="RdBu_r",
                    title=f"Matriz de Correlação — {name}",
                )

                results.append({
                    "title": f"🔗 Matriz de Correlação — {name}",
                    "type": "chart",
                    "content": fig
                })
                results.append({
                    "title": f"📝 Comentários — {name}",
                    "type": "text",
                    "content": (
                        "Resumo automático indisponível.\n\n"
                        "Correlações mais fortes encontradas localmente:\n- "
                        + "\n- ".join(top_pairs)
                    ),
                })

        return results

    # ============================================================
    # 🔹 MÉTODO ADICIONAL — PADRÕES CATEGÓRICOS
    # ============================================================
    def frequencies(self):
        """Analisa padrões de frequência em colunas categóricas."""
        results = []
        for name, df in self.dfs.items():
            cat_cols = df.select_dtypes(exclude="number").columns
            if len(cat_cols) == 0:
                continue

            for col in cat_cols:
                freq = df[col].value_counts().reset_index().head(10)
                freq.columns = [col, "Frequência"]

                # Gráfico local
                try:
                    fig = px.bar(freq, x=col, y="Frequência", title=f"Top categorias — {col} ({name})")
                    results.append({
                        "title": f"📊 Padrões de Frequência — {col} ({name})",
                        "type": "chart",
                        "content": fig
                    })
                except Exception:
                    continue

            # Interpretação via LLM
            try:
                prompt = (
                    f"Analise os padrões de frequência das variáveis categóricas do dataset '{name}'. "
                    f"Descreva quais categorias aparecem com maior e menor frequência, "
                    f"e o que isso pode sugerir sobre os dados. "
                    f"Resumo de frequências: {freq.to_dict()}."
                )
                commentary = self.llm.invoke(prompt).content
            except Exception:
                commentary = (
                    "Resumo automático indisponível. "
                    "Sugestão: observe as categorias dominantes — elas indicam concentração de registros "
                    "ou possíveis viéses de coleta."
                )

            results.append({
                "title": f"🧠 Interpretação Automática — {name}",
                "type": "text",
                "content": commentary
            })

        return results
