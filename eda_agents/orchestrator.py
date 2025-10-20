import os
from langchain_google_genai import ChatGoogleGenerativeAI
#from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import ChatPromptTemplate


from .analyst_agent import AnalystAgent
from .visualizer_agent import VisualizerAgent
from .pattern_agent import PatternAgent
from .anomaly_agent import AnomalyAgent
from .advisor_agent import AdvisorAgent
from memory import get_history

INTENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Você é um orquestrador de agentes EDA. Analise a pergunta e escolha SOMENTE UMA categoria entre:\n"
     " - 'analyst' (estatísticas, tipos, ausentes)\n"
     " - 'histogram' (gráficos de distribuição para variáveis numéricas)\n"
     " - 'boxplot' (gráficos de boxplots para variáveis numéricas)\n"
     " - 'barplot' (gráficos de barras para variáveis categóricas)\n"
     " - 'pie' (gráficos de pizza para variáveis categóricas com poucas categorias)\n"
     " - 'pattern' (correlações, frequências, clusters simples)\n"
     " - 'anomaly' (detecção de outliers)\n"
     " - 'advisor' (quando o usuário pedir conclusões gerais ou resumo das análises)\n"
     "Responda apenas com a palavra da categoria."),
    ("human", "{question}")
])

ANALYST_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Você é o Agente Analista. Foque em estatísticas descritivas, tipos e valores ausentes."),
    ("human", "{question}")
])
VISUAL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Você é o Agente Visualizador. Escolha gráficos adequados (histogramas, boxplots, barras, pizza)."),
    ("human", "{question}")
])
PATTERN_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Você é o Agente de Padrões. Foque em correlações numéricas e valores mais/menos frequentes."),
    ("human", "{question}")
])
ANOMALY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Você é o Agente de Anomalias. Foque em detecção de outliers com IQR."),
    ("human", "{question}")
])
ADVISOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Você é o Agente de Conclusões. Resuma descobertas e recomendações a partir do histórico de análises."),
    ("human", "{question}")
])

class Orchestrator:
    def __init__(self, dfs: dict, gemini_api_key: str):
        self.dfs = dfs
        self.analyst = AnalystAgent(dfs, gemini_api_key=gemini_api_key)
        self.visual = VisualizerAgent(dfs, gemini_api_key=gemini_api_key)
        self.patterns = PatternAgent(dfs, gemini_api_key=gemini_api_key)
        self.anomaly = AnomalyAgent(dfs, gemini_api_key=gemini_api_key)
        self.advisor = AdvisorAgent(gemini_api_key=gemini_api_key)

        model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=0, google_api_key=gemini_api_key)

    def classify(self, question: str) -> str:
        chain = INTENT_PROMPT | self.llm
        resp = chain.invoke({"question": question})
        return (resp.content or "").strip().lower()

    def answer(self, question: str, user="demo@local"):
        intent = self.classify(question)

        if "analyst" in intent:
            _ = (ANALYST_PROMPT | self.llm).invoke({"question": question})
            return {"agent": "AnalystAgent", "result": self.analyst.describe()}

        if "histogram" in intent:
            _ = (VISUAL_PROMPT | self.llm).invoke({"question": question})
            return {"agent": "VisualizerAgent", "result": self.visual.histograms()}

        if "boxplot" in intent:
            _ = (VISUAL_PROMPT | self.llm).invoke({"question": question})
            return {"agent": "VisualizerAgent", "result": self.visual.boxplots()}

        if "barplot" in intent:
            _ = (VISUAL_PROMPT | self.llm).invoke({"question": question})
            return {"agent": "VisualizerAgent", "result": self.visual.barplots()}

        if "pie" in intent:
            _ = (VISUAL_PROMPT | self.llm).invoke({"question": question})
            return {"agent": "VisualizerAgent", "result": self.visual.piecharts()}

        if "pattern" in intent:
            _ = (PATTERN_PROMPT | self.llm).invoke({"question": question})
            return {"agent": "PatternAgent", "result": self.patterns.correlations()}

        if "anomaly" in intent:
            _ = (ANOMALY_PROMPT | self.llm).invoke({"question": question})
            return {"agent": "AnomalyAgent", "result": self.anomaly.iqr_outliers()}

        if "advisor" in intent or "conclus" in question.lower() or "resum" in question.lower():
            _ = (ADVISOR_PROMPT | self.llm).invoke({"question": question})
            history = get_history(user, limit=20)
            return {"agent": "AdvisorAgent", "result": [self.advisor.summarize_history(history)]}

        return {
            "agent": "Unknown",
            "result": [{
                "title": "❓ Intenção não identificada",
                "type": "text",
                "content": "Reformule a pergunta."
            }]
        }
