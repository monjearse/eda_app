import pandas as pd
import plotly.express as px

class VisualizerAgent:
    def __init__(self, dfs):
        self.dfs = dfs

    def histograms(self):
        """Gera histogramas para variáveis numéricas + estatísticas + comentários"""
        results = []
        for name, df in self.dfs.items():
            numeric_cols = df.select_dtypes(include="number").columns
            for col in numeric_cols[:2]:  # limitar só a 2 variáveis para não sobrecarregar
                fig = px.histogram(df, x=col, nbins=30, title=f"Distribuição de {col}")
                results.append({"title": f"📊 Histograma — {col} ({name})", "type": "chart", "content": fig})

                desc = df[col].describe().to_frame().reset_index()
                results.append({"title": f"📋 Estatísticas rápidas — {col}", "type": "table", "content": desc})

                commentary = [
                    f"📌 A variável **{col}** tem média {df[col].mean():.2f}, mediana {df[col].median():.2f}, "
                    f"desvio padrão {df[col].std():.2f}.",
                    "Verifique caudas longas ou assimetria para potenciais outliers."
                ]
                results.append({"title": f"📝 Comentários — {col}", "type": "text", "content": "\n".join(commentary)})
        return results

    def barplots(self):
        """Gera gráficos de barras para variáveis categóricas + tabelas + comentários"""
        results = []
        for name, df in self.dfs.items():
            categorical_cols = df.select_dtypes(include="object").columns
            for col in categorical_cols[:1]:
                freq = df[col].value_counts().reset_index()
                freq.columns = [col, "Frequência"]

                fig = px.bar(freq, x=col, y="Frequência", title=f"Frequências de {col}")
                results.append({"title": f"📊 Frequências — {col} ({name})", "type": "chart", "content": fig})
                results.append({"title": f"📋 Frequências absolutas — {col}", "type": "table", "content": freq})

                commentary = [f"📌 Categoria mais frequente: **{freq.iloc[0][col]}** "
                              f"com {freq.iloc[0]['Frequência']} ocorrências."]
                if len(freq) > 1:
                    commentary.append(f"Segunda mais frequente: {freq.iloc[1][col]} "
                                      f"({freq.iloc[1]['Frequência']} ocorrências).")
                results.append({"title": f"📝 Comentários — {col}", "type": "text", "content": "\n".join(commentary)})
        return results

    def boxplots(self):
        """Gera boxplots para variáveis numéricas + estatísticas de outliers + comentários"""
        results = []
        for name, df in self.dfs.items():
            numeric_cols = df.select_dtypes(include="number").columns
            for col in numeric_cols[:2]:  # limitar só a 2 para não gerar overload
                # Gráfico boxplot
                fig = px.box(df, y=col, title=f"Boxplot de {col}")
                results.append({"title": f"📈 Boxplot — {col} ({name})", "type": "chart", "content": fig})

                # Estatísticas de outliers via IQR
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                outliers = df[(df[col] < lower) | (df[col] > upper)][col]

                stats = pd.DataFrame({
                    "Q1": [q1], "Q3": [q3], "IQR": [iqr],
                    "Limite Inferior": [lower], "Limite Superior": [upper],
                    "Qtd Outliers": [len(outliers)]
                })
                results.append({"title": f"📋 Estatísticas de outliers — {col}", "type": "table", "content": stats})

                # Comentários
                commentary = [f"📌 A variável **{col}** apresenta **{len(outliers)} outliers** detectados pelo método IQR."]
                if len(outliers) > 0:
                    commentary.append("Considere investigar valores extremos para possíveis erros ou casos raros.")
                results.append({"title": f"📝 Comentários — {col}", "type": "text", "content": "\n".join(commentary)})
        return results
