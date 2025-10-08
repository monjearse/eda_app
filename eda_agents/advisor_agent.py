import os
from langchain_google_genai import ChatGoogleGenerativeAI

class AdvisorAgent:
    """
    Agente responsável por gerar resumos gerais, recomendações,
    conclusões e perguntas sugeridas com base nas análises.
    Usa Gemini (modelo definido no .env).
    """

    def __init__(self, gemini_api_key):
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.2,
            google_api_key=gemini_api_key
        )

    def summarize(self, last_answer):
        """
        Gera resumo, recomendações e SEMPRE inclui perguntas sugeridas.
        """
        if not last_answer or "result" not in last_answer:
            return {
                "title": "📌 Resumo Geral e Recomendações",
                "type": "text",
                "content": (
                    "Nenhuma análise disponível ainda.\n\n"
                    "Perguntas sugeridas:\n"
                    "- Quais variáveis apresentam maior variabilidade?\n"
                    "- Existem correlações fortes entre variáveis numéricas?\n"
                    "- Quais categorias aparecem com mais frequência?"
                )
            }

        agent_name = last_answer.get("agent", "Agente desconhecido")
        blocks = last_answer["result"]

        prompt = f"""
        Você é um assistente de análise exploratória de dados.
        Baseado no resultado produzido pelo agente {agent_name}:

        {blocks}

        Gere:
        - Um resumo claro e curto (3 a 5 frases).
        - Entre 2 e 3 recomendações práticas para próximos passos.
        - 3 perguntas que o usuário pode fazer a seguir para entender melhor os dados.
        Sempre inclua a seção "Perguntas sugeridas:" no final.
        """

        try:
            commentary = self.llm.invoke(prompt).content.strip()
        except Exception:
            commentary = (
                "Resumo automático indisponível.\n\n"
                "Recomendações:\n"
                "- Continue explorando distribuições e correlações.\n"
                "- Analise outliers para identificar padrões atípicos.\n"
                "- Valide se existem valores ausentes relevantes.\n\n"
                "Perguntas sugeridas:\n"
                "- Quais variáveis apresentam maior variabilidade?\n"
                "- Existem correlações fortes entre variáveis numéricas?\n"
                "- Quais categorias aparecem com mais frequência?"
            )

        # Garantia de que sempre existam perguntas sugeridas
        if "Perguntas sugeridas:" not in commentary:
            commentary += (
                "\n\nPerguntas sugeridas:\n"
                "- Quais variáveis apresentam maior variabilidade?\n"
                "- Existem correlações fortes entre variáveis numéricas?\n"
                "- Quais categorias aparecem com mais frequência?"
            )

        return {
            "title": "📌 Resumo Geral, Recomendações e Perguntas Sugeridas",
            "type": "text",
            "content": commentary
        }

    def summarize_history(self, history):
        """
        Gera conclusões gerais com base no histórico de perguntas e respostas.
        """
        if not history:
            return {
                "title": "📌 Conclusões Gerais",
                "type": "text",
                "content": (
                    "Ainda não foram feitas análises suficientes para gerar conclusões.\n\n"
                    "Perguntas sugeridas:\n"
                    "- Qual tendência geral pode ser observada nos dados já analisados?\n"
                    "- Existem padrões repetidos ao longo do tempo?\n"
                    "- Quais insights podem ser aprofundados com novas análises?"
                )
            }

        hist_text = "\n".join([f"Pergunta: {q}\nResposta: {a}" for q, a, _ in history])

        prompt = f"""
        Você é um assistente de análise de dados.
        Aqui estão as perguntas e respostas anteriores do usuário:

        {hist_text}

        Com base nisso, escreva:
        - 3 a 5 conclusões objetivas já obtidas dos dados.
        - 2 recomendações práticas de próximos passos para análise.
        - 3 perguntas sugeridas para aprofundar a exploração dos dados.
        """

        try:
            resp = self.llm.invoke(prompt)
            content = resp.content.strip()
        except Exception as e:
            content = (
                f"Erro ao gerar conclusões: {e}\n\n"
                "Conclusões preliminares não disponíveis.\n\n"
                "Perguntas sugeridas:\n"
                "- Existe alguma variável com tendência temporal?\n"
                "- Quais relações ainda não foram exploradas?\n"
                "- Onde podem existir inconsistências nos dados?"
            )

        if "Perguntas sugeridas:" not in content:
            content += (
                "\n\nPerguntas sugeridas:\n"
                "- Existe alguma variável com tendência temporal?\n"
                "- Quais relações ainda não foram exploradas?\n"
                "- Onde podem existir inconsistências nos dados?"
            )

        return {
            "title": "📌 Conclusões Gerais",
            "type": "text",
            "content": content
        }
