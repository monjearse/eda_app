import os
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI

class AnomalyAgent:
    def __init__(self, dfs, gemini_api_key):
        self.dfs = dfs
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")
        self.llm = ChatGoogleGenerativeAI(
            model=model_name, temperature=0.2, google_api_key=gemini_api_key
        )

    def iqr_outliersOld(self):
        results = []
        for name, df in self.dfs.items():
            numeric_cols = df.select_dtypes(include="number").columns
            for col in numeric_cols:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                outliers = df[(df[col] < lower) | (df[col] > upper)][col]

                summary = {
                    "q1": q1, "q3": q3, "iqr": iqr,
                    "lower": lower, "upper": upper,
                    "outliers": len(outliers)
                }
                results.append({
                    "title": f"⚠️ Outliers detectados — {col} ({name})",
                    "type": "json",
                    "content": summary
                })

            prompt = f"Faça um comentário sobre os outliers detectados no dataset {name}."
            try:
                commentary = self.llm.invoke(prompt).content
            except Exception:
                commentary = (
                    "Resumo automático indisponível. "
                    "Sugestão: variáveis com muitos outliers merecem investigação "
                    "pois podem indicar erros de entrada ou eventos raros."
                )

            results.append({
                "title": f"📝 Comentários — {name}",
                "type": "text",
                "content": commentary
            })
        return results

    def iqr_outliers(self):
        import plotly.express as px
        results = []

        for name, df in self.dfs.items():
            # 🔹 Tenta usar o LLM como primeira via
            try:
                numeric_cols = df.select_dtypes(include="number").columns
                prompt = (
                    f"Você é um especialista em análise de dados. "
                    f"Avalie o dataset '{name}' e identifique quais variáveis numéricas "
                    f"apresentam outliers com base no método IQR (Interquartile Range). "
                    f"Explique resumidamente, em tom humano, "
                    f"quais variáveis são mais críticas e o que isso pode indicar. "
                    f"Dados estatísticos iniciais: {df.describe(include='number').to_dict()}."
                )

                commentary = self.llm.invoke(prompt).content

                results.append({
                    "title": f"🧠 Interpretação Automática — {name}",
                    "type": "text",
                    "content": commentary
                })

                # 🔹 Boxplots automáticos (suporte visual)
                numeric_cols = df.select_dtypes(include="number").columns
                for col in numeric_cols:
                    try:
                        fig = px.box(df, y=col, title=f"Boxplot de {col} — {name}")
                        results.append({
                            "title": f"📊 Boxplot — {col} ({name})",
                            "type": "chart",
                            "content": fig
                        })
                    except Exception:
                        pass

            except Exception:
                # 🔹 Fallback local (caso LLM falhe)
                numeric_cols = df.select_dtypes(include="number").columns
                resumo_geral = []

                for col in numeric_cols:
                    q1 = df[col].quantile(0.25)
                    q3 = df[col].quantile(0.75)
                    iqr = q3 - q1
                    lower = q1 - 1.5 * iqr
                    upper = q3 + 1.5 * iqr
                    outliers = df[(df[col] < lower) | (df[col] > upper)][col]

                    resumo = (
                        f"A variável **{col}** possui **{len(outliers)} outlier(s)**. "
                        f"IQR = {iqr:.2f}, limites [{lower:.2f}, {upper:.2f}]."
                    )
                    resumo_geral.append(resumo)

                    try:
                        fig = px.box(df, y=col, title=f"Boxplot de {col} — {name}")
                        results.append({
                            "title": f"📊 Boxplot — {col} ({name})",
                            "type": "chart",
                            "content": fig
                        })
                    except Exception:
                        pass

                results.append({
                    "title": f"⚠️ Resumo de Outliers — {name}",
                    "type": "text",
                    "content": "\n".join(resumo_geral)
                })

                results.append({
                    "title": f"📝 Comentários — {name}",
                    "type": "text",
                    "content": (
                        "Resumo automático indisponível. "
                        "Variáveis com muitos outliers podem indicar erros de medição, "
                        "valores atípicos ou fenômenos raros que merecem investigação."
                    )
                })

        return results
