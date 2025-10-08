import os
import plotly.express as px
from langchain_google_genai import ChatGoogleGenerativeAI

class VisualizerAgent:
    """
    Agente responsável por gerar representações gráficas dos dados:
    - histogramas, boxplots, barras e pizzas.
    Sempre prioriza o uso do LLM (Gemini) para interpretação textual,
    com fallback local quando o modelo não está disponível.
    """

    def __init__(self, dfs, gemini_api_key):
        self.dfs = dfs
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.2,
            google_api_key=gemini_api_key
        )

    # ============================================================
    # 🔹 HISTOGRAMAS
    # ============================================================
    def histograms(self):
        results = []
        for name, df in self.dfs.items():
            numeric_cols = df.select_dtypes(include="number").columns

            # Gráficos locais
            for col in numeric_cols:
                try:
                    fig = px.histogram(df, x=col, title=f"Distribuição de {col} — {name}")
                    results.append({
                        "title": f"📈 Distribuição — {col} ({name})",
                        "type": "chart",
                        "content": fig
                    })
                except Exception:
                    continue

            # Interpretação automática via LLM
            try:
                prompt = (
                    f"Você é um analista de dados. Analise os histogramas das variáveis numéricas "
                    f"do dataset '{name}' e descreva brevemente padrões visíveis: assimetrias, "
                    f"dispersões e concentrações. Dados de apoio: {df.describe(include='number').to_dict()}."
                )
                commentary = self.llm.invoke(prompt).content
            except Exception:
                commentary = (
                    "Resumo automático indisponível. "
                    "Sugestão: observe os histogramas para identificar distribuições assimétricas "
                    "e picos de frequência, que indicam concentração de valores ou outliers."
                )

            results.append({
                "title": f"🧠 Interpretação Automática — {name}",
                "type": "text",
                "content": commentary
            })

        return results

    # ============================================================
    # 🔹 BOXPLOTS
    # ============================================================
    def boxplots(self):
        """Gera boxplots para variáveis numéricas com interpretação automática via LLM."""
        results = []
        for name, df in self.dfs.items():
            numeric_cols = df.select_dtypes(include="number").columns

            # Geração dos gráficos
            for col in numeric_cols:
                try:
                    fig = px.box(df, y=col, title=f"Boxplot de {col} — {name}")
                    results.append({
                        "title": f"📊 Boxplot — {col} ({name})",
                        "type": "chart",
                        "content": fig
                    })
                except Exception:
                    continue

            # Interpretação com LLM prioritário
            try:
                prompt = (
                    f"Analise os boxplots das variáveis numéricas do dataset '{name}'. "
                    f"Explique em linguagem natural quais variáveis apresentam maior dispersão, "
                    f"assimetria ou outliers significativos. Use um tom analítico e direto. "
                    f"Dados estatísticos: {df.describe(include='number').to_dict()}."
                )
                commentary = self.llm.invoke(prompt).content
            except Exception:
                commentary = (
                    "Resumo automático indisponível. "
                    "Sugestão: observe variáveis com grande dispersão nos boxplots — "
                    "elas indicam variabilidade alta ou presença de outliers."
                )

            results.append({
                "title": f"🧠 Interpretação Automática — {name}",
                "type": "text",
                "content": commentary
            })

        return results

    # ============================================================
    # 🔹 GRÁFICOS DE BARRAS
    # ============================================================
    def barplots(self):
        """Gera gráficos de barras para variáveis categóricas com interpretação automática via LLM."""
        results = []
        for name, df in self.dfs.items():
            cat_cols = df.select_dtypes(exclude="number").columns

            for col in cat_cols:
                try:
                    top_cat = df[col].value_counts().reset_index().head(10)
                    top_cat.columns = [col, "Frequência"]
                    fig = px.bar(top_cat, x=col, y="Frequência", title=f"Top categorias — {col} ({name})")
                    results.append({
                        "title": f"📊 Categorias mais frequentes — {col} ({name})",
                        "type": "chart",
                        "content": fig
                    })
                except Exception:
                    continue

            # LLM interpretação
            try:
                prompt = (
                    f"Analise as distribuições categóricas do dataset '{name}'. "
                    f"Identifique categorias dominantes, raras e possíveis desequilíbrios "
                    f"de frequência. Dados de apoio: {df.describe(include='object').to_dict()}."
                )
                commentary = self.llm.invoke(prompt).content
            except Exception:
                commentary = (
                    "Resumo automático indisponível. "
                    "Sugestão: observe categorias dominantes e pouco representadas — "
                    "elas indicam concentração de registros ou casos raros."
                )

            results.append({
                "title": f"🧠 Interpretação Automática — {name}",
                "type": "text",
                "content": commentary
            })

        return results

    # ============================================================
    # 🔹 GRÁFICOS DE PIZZA
    # ============================================================
    def piecharts(self):
        """Gera gráficos de pizza para variáveis categóricas com poucas categorias (<=6)."""
        results = []
        for name, df in self.dfs.items():
            cat_cols = df.select_dtypes(exclude="number").columns
            for col in cat_cols:
                try:
                    counts = df[col].value_counts()
                    if 2 <= len(counts) <= 6:
                        fig = px.pie(values=counts.values, names=counts.index,
                                     title=f"Distribuição de {col} ({name})")
                        results.append({
                            "title": f"🥧 Distribuição (pizza) — {col} ({name})",
                            "type": "chart",
                            "content": fig
                        })
                except Exception:
                    continue

            # Interpretação automática
            try:
                prompt = (
                    f"Analise os gráficos de pizza do dataset '{name}'. "
                    f"Explique brevemente o equilíbrio entre categorias, e destaque se há predominância "
                    f"de alguma delas. Dados categóricos: {df.describe(include='object').to_dict()}."
                )
                commentary = self.llm.invoke(prompt).content
            except Exception:
                commentary = (
                    "Resumo automático indisponível. "
                    "Sugestão: observe gráficos com categorias dominantes — "
                    "elas podem indicar concentração de casos ou viés de amostragem."
                )

            results.append({
                "title": f"🧠 Interpretação Automática — {name}",
                "type": "text",
                "content": commentary
            })

        return results
