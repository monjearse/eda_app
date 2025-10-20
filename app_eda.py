import os, sys
import streamlit as st
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv

# 🆕 Suporte a URL/streaming/chunks
import requests
import tempfile

from io import StringIO

sys.path.append(os.path.dirname(__file__))
from utils_eda import read_any
from eda_agents.orchestrator import Orchestrator
from memory import init_memory, save_qa, get_history_filtered, get_all_users
from langchain_google_genai import ChatGoogleGenerativeAI

# ========== Boot ==========
load_dotenv()
gemini_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")

init_memory()
st.set_page_config(page_title="EDA por Agentes", layout="wide")

# Estado da sessão
if "dfs" not in st.session_state:
    st.session_state["dfs"] = None
if "user" not in st.session_state:
    st.session_state["user"] = "demo@local"
if "general_summary" not in st.session_state:
    st.session_state["general_summary"] = None

# ========== Sidebar ==========
with st.sidebar:
    st.title("⚙️ Configurações Rápidas")
    st.session_state["user"] = st.text_input("Usuário ativo", st.session_state["user"])

    st.divider()
    st.markdown("""
    ### 🎓 Curso
    **Agentes Inteligentes com IA Generativa**  
    I2A2 – Institut d'Intelligence Artificielle Appliquée
    """)

    st.divider()
    st.header("👥 Agentes do Processo")
    st.markdown("""
    - 📊 **AnalystAgent** → Estatísticas descritivas, tipos, valores ausentes  
    - 📈 **VisualizerAgent** → Gráficos (histogramas, boxplots, barras)  
    - 🔗 **PatternAgent** → Correlações, padrões e frequências  
    - ⚠️ **AnomalyAgent** → Detecção de outliers (IQR)  
    - 🧠 **AdvisorAgent** → Resumos gerais e conclusões automáticas
    """)

    st.divider()
    st.markdown("Desenvolvido por [Arsénio António Monjane](https://github.com/monjearse) com [LangChain](https://langchain.com/) e [Streamlit](https://streamlit.io/).")

st.title("🔎 EDA baseada em Agentes (LangChain + Streamlit)")

# ========== Tabs ==========
tabs = st.tabs(["📂 Processamento", "📊 Resumo Geral", "🗂 Histórico", "⚙️ Configurações"])

# ===== Funções auxiliares (modo URL/chunks) =====
def download_to_temp(url: str, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """
    Faz download do ficheiro por streaming para um arquivo temporário e retorna o caminho.
    Evita estourar memória durante o download.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            for b in r.iter_content(chunk_size=chunk_bytes):
                if b:  # ignora keep-alive
                    tmp.write(b)
        return tmp.name

# ========== Aba Processamento ==========
with tabs[0]:
    st.subheader("📂 Carregamento e Perguntas sobre os Dados")
    col_left, col_right = st.columns([1, 2])

    with col_left:
        #st.markdown("### 🗂️ Fonte dos dados")
        via_url = st.toggle("Carregar ficheiro via URL (para >200 MB)", value=False)

        if not via_url:
            # ------- Upload normal (≤ 200 MB por ficheiro na Streamlit Cloud) -------
            uploaded = st.file_uploader(
                "Carregue CSVs ou ZIPs (máx. 200 MB cada)",
                type=["csv", "zip"],
                accept_multiple_files=True
            )

            if uploaded:
                dfs = read_any(uploaded)
                st.session_state["dfs"] = dfs
                st.success(f"{len(dfs)} dataset(s) carregado(s): {list(dfs.keys())}")

                # Resumo textual inicial
                from eda_agents.advisor_agent import AdvisorAgent
                advisor = AdvisorAgent(gemini_api_key=gemini_key)
                st.session_state["general_summary"] = advisor.summarize({
                    "agent": "System",
                    "result": [f"Dados carregados: {list(dfs.keys())}"]
                })

                # Resumo tabular e visual rápidos
                summary_tables, summary_charts = [], []
                for name, df in dfs.items():
                    dims = pd.DataFrame({
                        "Dataset": [name],
                        "Linhas": [df.shape[0]],
                        "Colunas": [df.shape[1]],
                        "Variáveis Numéricas": [len(df.select_dtypes(include="number").columns)],
                        "Variáveis Categóricas": [len(df.select_dtypes(exclude="number").columns)]
                    })
                    summary_tables.append(dims)

                    num_cols = df.select_dtypes(include="number").columns
                    if len(num_cols) > 0:
                        fig_num = px.histogram(df, x=num_cols[0], nbins=30,
                                               title=f"Distribuição inicial — {name} [{num_cols[0]}]")
                        summary_charts.append(fig_num)

                    cat_cols = df.select_dtypes(exclude="number").columns
                    if len(cat_cols) > 0:
                        top_cat = df[cat_cols[0]].value_counts().reset_index().head(10)
                        top_cat.columns = [cat_cols[0], "Frequência"]
                        fig_cat = px.bar(top_cat, x=cat_cols[0], y="Frequência",
                                         title=f"Categorias mais frequentes — {name} [{cat_cols[0]}]")
                        summary_charts.append(fig_cat)

                st.session_state["general_summary_tables"] = summary_tables
                st.session_state["general_summary_charts"] = summary_charts
                st.info("✅ Dados carregados. Um resumo geral foi gerado automaticamente (veja na aba **Resumo Geral**). Agora pode começar a fazer perguntas.")

            elif st.session_state["dfs"] is not None:
                dfs = st.session_state["dfs"]
                st.info(f"Já existem {len(dfs)} dataset(s) carregado(s): {list(dfs.keys())}")
                if st.button("🧹 Limpar datasets carregados"):
                    st.session_state["dfs"] = None
                    st.session_state["general_summary"] = None
                    st.success("Datasets e resumo geral removidos. Faça upload novamente para continuar.")
            else:
                st.info("Faça upload de pelo menos um CSV para começar.")

        else:
            # ------- Modo via URL com processamento incremental (recomendado) -------
            st.markdown("Cole o link direto do ficheiro **CSV** (ex.: `https://.../dados.csv`).")
            url = st.text_input("URL do ficheiro")

            col_dl, col_cfg = st.columns([1, 1])
            with col_dl:
                processar = st.button("📥 Carregar e processar via URL",
                                      help="Carrega e processa o ficheiro via urlpor partes. Evita estourar memória.")
                
            with col_cfg:
                chunksize_input = st.number_input(
                    "Tamanho do chunk (linhas)",
                    min_value=10_000,
                    max_value=500_000,
                    step=10_000,
                    value=50_000,  # ✅ recomendação aplicada
                    help="Cada bloco é processado separadamente. Evita estourar memória."
                )

            if processar and url:
                try:
                    with st.spinner("🔄 A descarregar ficheiro (streaming)..."):
                        tmp_path = download_to_temp(url)

                    st.info("🧠 A processar por partes com o Orchestrator (sem exibir parciais)…")
                    orch = Orchestrator({}, gemini_api_key=gemini_key)

                    total_linhas = 0
                    total_chunks = 0

                    # Barra de progresso simples (desacoplada do nº real de chunks)
                    progress = st.progress(0)
                    progress_step = 0

                    # Processamento incremental: NÃO concatena, NÃO renderiza parciais
                    for i, chunk in enumerate(pd.read_csv(tmp_path, chunksize=int(chunksize_input))):
                        total_chunks += 1
                        total_linhas += len(chunk)

                        # Chamada ao orquestrador por chunk
                        ans = orch.answer("Gerar resumo automático deste chunk", df=chunk)

                        # Registo no histórico (sem renderizar outputs)
                        save_qa(st.session_state["user"], f"Chunk {i+1}", str(ans))

                        # Atualiza progress bar (progresso heurístico)
                        progress_step = min(progress_step + 0.05, 0.95)
                        progress.progress(progress_step)

                    # Consolidação final pelo AdvisorAgent
                    from eda_agents.advisor_agent import AdvisorAgent
                    advisor = AdvisorAgent(gemini_api_key=gemini_key)
                    resumo_final = advisor.summarize({
                        "agent": "System",
                        "result": [f"{total_chunks} chunks processados ({total_linhas} linhas no total)"]
                    })
                    st.session_state["general_summary"] = resumo_final
                    progress.progress(1.0)

                    st.success(f"✅ Processamento completo! {total_chunks} partes, {total_linhas} linhas no total.")
                    st.markdown("---")
                    st.subheader("🧠 Resumo Geral Final")
                    st.write(resumo_final["content"])

                    # Como não concatenamos, mantemos um placeholder leve em dfs
                    st.session_state["dfs"] = {"via_url.csv": pd.DataFrame()}  # placeholder leve

                except Exception as e:
                    st.error(f"Erro ao descarregar/processar o ficheiro: {e}")

        # ===== Entrada de pergunta =====
        if st.session_state["dfs"] is not None:
            st.session_state["user"] = st.text_input("Identificador do usuário", st.session_state["user"])

            def limpar_pergunta():
                st.session_state["question_input"] = ""

            q = st.text_input(
                "Ex.: 'Quais variáveis têm outliers?', 'Mostre as distribuições', 'Correlação entre colunas'.",
                key="question_input"
            )

            col_q1, col_q2 = st.columns([4, 1])
            with col_q1:
                ask = st.button("Responder")
            with col_q2:
                st.button("❌", on_click=limpar_pergunta)
        else:
            q, ask = None, False

    # ===== Coluna direita: respostas =====
    with col_right:
        if st.session_state["dfs"] is None:
            st.warning("📂 Carregue datasets antes de fazer perguntas.")
        else:
            orch = Orchestrator(st.session_state["dfs"], gemini_api_key=gemini_key)

            if ask and q:
                with st.spinner("⏳ Processando sua pergunta..."):
                    ans = orch.answer(q)
                save_qa(st.session_state["user"], q, str(ans))
                st.session_state["last_answer"] = ans
                st.markdown(f"**Agente chamado:** `{ans.get('agent', 'Desconhecido')}`")

                if not ans.get("result"):
                    st.warning("Nenhum resultado detalhado foi retornado pelo agente.")

                blocks = ans.get("result", [])
                for block in blocks:
                    st.subheader(block.get("title", ""))
                    st.markdown("---")

                    btype = block.get("type", "text")
                    content = block.get("content")
                    if btype == "chart":
                        st.plotly_chart(content, use_container_width=True)
                    elif btype == "json":
                        st.json(content)
                    elif btype == "table":
                        st.dataframe(content, use_container_width=True)
                    else:
                        st.write(content)
            else:
                with st.spinner("⏳ Aguardando sua pergunta..."):
                    st.info("Digite sua pergunta no painel à esquerda e clique em **Responder** para começar a análise.")

# ========== Aba Resumo Geral ==========
with tabs[1]:
    if st.session_state.get("general_summary"):
        block = st.session_state["general_summary"]
        content = block["content"]

        if "Perguntas sugeridas:" in content:
            parts = content.split("Perguntas sugeridas:")
            resumo = parts[0].strip()
            perguntas = parts[1].strip().split("\n")

            st.write(resumo)
            st.subheader("❓ Perguntas sugeridas")
            for p in perguntas:
                if p.strip() and any(ch.isalpha() for ch in p):
                    st.markdown(f"- {p.strip()}")
        else:
            st.write(content)
    else:
        st.info("Ainda não foi gerado nenhum resumo geral. Faça upload de dados primeiro.")

# ========== Aba Histórico ==========
with tabs[2]:
    st.subheader("🗂 Histórico completo")

    from datetime import date
    all_users = get_all_users()
    user_options = ["(Todos)"] + all_users

    hoje = date.today()
    col1, col2, col3 = st.columns([1, 1, 1])
    filter_user = col1.selectbox("Usuário", user_options, index=0)
    start_date = col2.date_input("Data inicial", value=hoje)
    end_date = col3.date_input("Data final", value=hoje)

    start_str = start_date.isoformat() if start_date else None
    end_str = end_date.isoformat() if end_date else None

    rows = get_history_filtered(
        user=None if filter_user == "(Todos)" else filter_user,
        start_date=start_str,
        end_date=end_str,
        limit=500
    )

    if rows:
        df_hist = pd.DataFrame(rows, columns=["Usuário", "Pergunta", "Resposta", "Data"])
        total_registros = len(df_hist)
        itens_por_pagina = 10
        total_paginas = (total_registros - 1) // itens_por_pagina + 1

        if "pagina_hist" not in st.session_state:
            st.session_state["pagina_hist"] = 1

        col_a, col_b, col_c = st.columns([1, 2, 1])
        with col_a:
            if st.button("⬅️ Anterior") and st.session_state["pagina_hist"] > 1:
                st.session_state["pagina_hist"] -= 1
        with col_c:
            if st.button("Próxima ➡️") and st.session_state["pagina_hist"] < total_paginas:
                st.session_state["pagina_hist"] += 1

        inicio = (st.session_state["pagina_hist"] - 1) * itens_por_pagina
        fim = inicio + itens_por_pagina
        df_pagina = df_hist.iloc[inicio:fim]

        st.dataframe(df_pagina, use_container_width=True, height=400)
        st.caption(
            f"📄 Página {st.session_state['pagina_hist']} de {total_paginas}  "
            f"• Mostrando registros de {start_date} a {end_date}  "
            f"• Total de {total_registros} registros."
        )
    else:
        st.info(f"Nenhum histórico encontrado para o período selecionado ({start_date}).")

# ========== Aba Configurações ==========
with tabs[3]:
    st.subheader("⚙️ Configurações")
    st.session_state["user"] = st.text_input("✍️ Identificador padrão do usuário", st.session_state["user"])

    if "response_height" not in st.session_state:
        st.session_state["response_height"] = 600
    st.session_state["response_height"] = st.slider(
        "Altura da área de respostas (px)",
        min_value=300,
        max_value=1000,
        step=100,
        value=st.session_state["response_height"]
    )

    st.subheader("📌 Sobre a solução")
    st.caption(
        "**Aplicação para Análise Exploratória de Dados (EDA)**:"
        "carregue arquivos CSV e faça perguntas em linguagem natural."
        "Agentes especializados geram estatísticas, gráficos, padrões,"
        "anomalias e um agente de recomendações produz resumos"
        "e conclusões automáticas."
    )

    st.write(f"🔑 Modelo configurado: `{os.getenv('GEMINI_MODEL', 'gemini-2.0-flash-exp')}`")
    st.write("A chave da API é carregada automaticamente do arquivo `.env`.")

    st.divider()
    st.subheader("🔎 Teste da Chave Gemini")
    if st.button("Testar chave da API"):
        try:
            model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=0,
                google_api_key=gemini_key
            )
            with st.spinner("⏳ Validando chave..."):
                resp = llm.invoke("Responda apenas com: OK")
            st.success(f"✅ Chave válida! Resposta: {resp.content}")
        except Exception as e:
            st.error(f"❌ Erro ao validar chave: {e}")
