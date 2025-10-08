import pandas as pd
import plotly.express as px

class PatternAgent:
    def __init__(self, dfs):
        self.dfs = dfs

    def correlations(self):
        results = []
        for name, df in self.dfs.items():
            try:
                # Calcular correlação numérica
                corr = df.corr(numeric_only=True)

                # Bloco 1: tabela
                results.append({
                    "title": f"🔗 Correlações — {name}",
                    "type": "table",
                    "content": corr.reset_index().rename(columns={"index": "Variável"})
                })

                # Bloco 2: heatmap
                fig = px.imshow(corr, text_auto=True, aspect="auto",
                                title=f"Mapa de calor das correlações — {name}")
                results.append({
                    "title": f"🌡️ Heatmap — {name}",
                    "type": "chart",
                    "content": fig
                })

                # Bloco 3: comentários
                commentary = []
                commentary.append("📌 Observações sobre as correlações:")
                high_corr = corr.abs().unstack().sort_values(ascending=False)
                high_corr = high_corr[high_corr < 1].head(3)  # top 3 correlações fortes
                if not high_corr.empty:
                    for (var1, var2), val in high_corr.items():
                        commentary.append(f"- {var1} e {var2} apresentam correlação {val:.2f}.")
                else:
                    commentary.append("- Não foram identificadas correlações relevantes.")
                
                results.append({
                    "title": f"📝 Comentários — {name}",
                    "type": "text",
                    "content": "\n".join(commentary)
                })

            except Exception as e:
                results.append({
                    "title": f"Erro ao calcular correlações — {name}",
                    "type": "text",
                    "content": str(e)
                })

        return results
