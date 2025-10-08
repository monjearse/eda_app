import pandas as pd
import plotly.express as px

class AnomalyAgent:
    def __init__(self, dfs):
        self.dfs = dfs

    def iqr_outliers(self):
        results = []
        for name, df in self.dfs.items():
            try:
                summary_rows = []
                numeric_cols = df.select_dtypes(include="number").columns

                for col in numeric_cols:
                    q1 = df[col].quantile(0.25)
                    q3 = df[col].quantile(0.75)
                    iqr = q3 - q1
                    lower = q1 - 1.5 * iqr
                    upper = q3 + 1.5 * iqr
                    outliers = df[(df[col] < lower) | (df[col] > upper)][col]

                    summary_rows.append({
                        "Variável": col,
                        "Q1": q1,
                        "Q3": q3,
                        "IQR": iqr,
                        "Limite Inferior": lower,
                        "Limite Superior": upper,
                        "Qtd Outliers": len(outliers)
                    })

                # Bloco 1: tabela
                summary_df = pd.DataFrame(summary_rows)
                results.append({
                    "title": f"⚠️ Outliers detectados — {name}",
                    "type": "table",
                    "content": summary_df
                })

                # Bloco 2: boxplot (primeira variável numérica só como exemplo)
                if len(numeric_cols) > 0:
                    col = numeric_cols[0]
                    fig = px.box(df, y=col, title=f"📈 Boxplot — {col}")
                    results.append({
                        "title": f"📈 Boxplot — {col}",
                        "type": "chart",
                        "content": fig
                    })

                # Bloco 3: comentários
                commentary = ["📌 Observações sobre os outliers:"]
                if not summary_df.empty:
                    top_outliers = summary_df.sort_values("Qtd Outliers", ascending=False).head(3)
                    for _, row in top_outliers.iterrows():
                        commentary.append(
                            f"- A variável **{row['Variável']}** apresenta **{row['Qtd Outliers']} outliers**."
                        )
                else:
                    commentary.append("- Não foram detectados outliers relevantes.")

                results.append({
                    "title": f"📝 Comentários — {name}",
                    "type": "text",
                    "content": "\n".join(commentary)
                })

            except Exception as e:
                results.append({
                    "title": f"Erro ao detectar outliers — {name}",
                    "type": "text",
                    "content": str(e)
                })

        return results
