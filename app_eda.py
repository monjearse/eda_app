import os, sys
import streamlit as st
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv

sys.path.append(os.path.dirname(__file__))
from utils_eda import read_any
from eda_agents.orchestrator import Orchestrator
from memory import init_memory, save_qa, get_history, get_history_filtered, get_all_users
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()
#gemini_key = os.getenv("GEMINI_API_KEY")
gemini_key = st.secrets["GEMINI_API_KEY"]

init_memory()

st.set_page_config(page_title="EDA por Agentes", layout="wide")
# Estado para manter datasets carregados e usuário
if "dfs" not in st.session_state:
    st.session_state["dfs"] = None
if "user" not in st.session_state:
    st.session_state["user"] = "demo@local"
if "general_summary" not in st.session_state:
    st.session_state["general_summary"] = None

# --- Sidebar ---
# --- Sidebar ---
with st.sidebar:
    st.title("⚙️ Configurações Rápidas")

    # Usuário ativo

    st.session_state["user"] = st.text_input("Usuário ativo", st.session_state["user"])


  
    st.divider()
    with st.sidebar:
    # Logo do I2A2
        #st.image("images/i2a2_logo.png", use_container_width=True)

        st.markdown("""
        ### 🎓 Curso
        **Agentes Inteligentes com IA Generativa**  
        I2A2 – Institut d'Intelligence Artificielle Appliquée
        """)
    
    st.divider()

#     # Âmbito do projeto
#  # Âmbito / Descrição da solução
#     st.subheader("📌 Sobre a solução")
#     st.caption(

#         "**Aplicação para Análise Exploratória de Dados (EDA)**:" 
#         "carregue arquivos CSV e faça perguntas em linguagem natural."
#         "Agentes especializados geram estatísticas, gráficos, padrões," 
#         "anomalias e um agente de recomendações produz resumos"
#         "e conclusões automáticas."
#     )

    # Listagem de agentes incluídos

    st.header("👥 Agentes do Processo")
    st.markdown("""
    - 📊 **AnalystAgent** → Estatísticas descritivas, tipos, valores ausentes
    - 📈 **VisualizerAgent** → Gráficos (histogramas, boxplots, barras)
    - 🔗 **PatternAgent** → Correlações, padrões e frequências
    - ⚠️ **AnomalyAgent** → Detecção de outliers com IQR
    - 🧠 **AdvisorAgent** → Resumos gerais e conclusões automáticas
    """)


    st.divider()

    # Chave de API
    st.markdown("Desenvolvido por [Arsénio António Monjane](https://github.com/monjearse) com o uso de [LangChain](https://langchain.com/) e [Streamlit](https://streamlit.io/).")

st.title("🔎 EDA baseada em Agentes (LangChain + Streamlit)")





# Tabs principais
tabs = st.tabs(["📂 Processamento", "📊 Resumo Geral", "🗂 Histórico", "⚙️ Configurações"])

# --- Aba Processamento ---
with tabs[0]:
    st.subheader("📂 Carregamento e Perguntas sobre os Dados")

    col_left, col_right = st.columns([1, 2])

    # --- Coluna esquerda ---
    with col_left:
        uploaded = st.file_uploader("Carregue CSVs ou ZIPs", type=["csv","zip"], accept_multiple_files=True)

        # if uploaded:
        #     dfs = read_any(uploaded)
        #     st.session_state["dfs"] = dfs
        #     st.success(f"{len(dfs)} dataset(s) carregado(s): {list(dfs.keys())}")

        #     # Gera resumo geral automaticamente
        #     from eda_agents.advisor_agent import AdvisorAgent
        #     advisor = AdvisorAgent(gemini_api_key=gemini_key)
        #     st.session_state["general_summary"] = advisor.summarize({
        #         "agent": "System",
        #         "result": [f"Dados carregados: {list(dfs.keys())}"]
        #     })

        #     st.info("✅ Dados carregados. Um resumo geral foi gerado automaticamente (veja na aba **Resumo Geral**). Agora pode começar a fazer perguntas.")

        if uploaded:
            dfs = read_any(uploaded)
            st.session_state["dfs"] = dfs
            st.success(f"{len(dfs)} dataset(s) carregado(s): {list(dfs.keys())}")

            # --- Resumo textual pelo AdvisorAgent ---
            from eda_agents.advisor_agent import AdvisorAgent
            advisor = AdvisorAgent(gemini_api_key=gemini_key)
            st.session_state["general_summary"] = advisor.summarize({
                "agent": "System",
                "result": [f"Dados carregados: {list(dfs.keys())}"]
            })

            # --- Resumo tabular e visual ---
            import pandas as pd
            import plotly.express as px

            summary_tables = []
            summary_charts = []

            for name, df in dfs.items():
                # Dimensões
                dims = pd.DataFrame({
                    "Dataset": [name],
                    "Linhas": [df.shape[0]],
                    "Colunas": [df.shape[1]],
                    "Variáveis Numéricas": [len(df.select_dtypes(include="number").columns)],
                    "Variáveis Categóricas": [len(df.select_dtypes(exclude="number").columns)]
                })
                summary_tables.append(dims)

                # Gráfico rápido: se tiver coluna numérica
                num_cols = df.select_dtypes(include="number").columns
                if len(num_cols) > 0:
                    fig_num = px.histogram(df, x=num_cols[0], nbins=30,
                                        title=f"Distribuição inicial — {name} [{num_cols[0]}]")
                    summary_charts.append(fig_num)

                # Gráfico rápido: se tiver coluna categórica
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
            # Botão para resetar datasets carregados
            if st.button("🧹 Limpar datasets carregados"):
                st.session_state["dfs"] = None
                st.session_state["general_summary"] = None # 🔄 reseta também o resumo geral
                st.success("Datasets e resumo geral removidos. Faça upload novamente para continuar.")

        else:
            st.info("Faça upload de pelo menos um CSV para começar.")

        # Inputs de usuário e pergunta
        # if st.session_state["dfs"] is not None:
        #     st.session_state["user"] = st.text_input("Identificador do usuário", st.session_state["user"])
        #     q = st.text_input("Ex.: 'Quais variáveis têm outliers?', 'Mostre as distribuições', 'Correlação entre colunas'.")

        #     col_q1, col_q2 = st.columns([4,1])
        #     with col_q1:
        #         ask = st.button("Responder")
        #     with col_q2:
        #         if st.button("❌"):
        #             q = ""
        #             st.session_state["q_temp"] = ""

        #     # Mini-histórico do usuário
        #     # st.divider()
        #     # st.subheader("🕑 Últimas perguntas do usuário")
        #     # history = get_history(st.session_state["user"], limit=3)
        #     # if history:
        #     #     for q_text, a_text, created_at in history:
        #     #         with st.expander(f"{created_at} — {q_text[:60]}"):
        #     #             st.write(a_text)
        #     # else:
        #     #     st.info("Nenhum histórico disponível para este usuário.")
                
        #     # Perguntas sugeridas dinamicamente pelo AdvisorAgent
        #     st.divider()
        #     #st.subheader("💡 Perguntas sugeridas pelo AdvisorAgent")
        #     st.markdown("""
        #         ### 💡 Perguntas sugeridas pelo AdvisorAgent
        #         """)

        #     if st.session_state.get("general_summary"):
        #         block = st.session_state["general_summary"]
        #         content = block["content"]

        #         if "Perguntas sugeridas:" in content:
        #             parts = content.split("Perguntas sugeridas:")
        #             perguntas = parts[1].strip().split("\n")

        #             # Botões de sugestão
        #             for p in perguntas:
        #                 pergunta = p.strip()
        #                 if pergunta and any(ch.isalpha() for ch in pergunta):
        #                     if st.button(pergunta):
        #                         # Preenche automaticamente a caixa de pergunta
        #                         st.session_state["q_temp"] = pergunta
        #                         st.experimental_rerun()
        #         else:
        #             st.info("Nenhuma sugestão gerada pelo AdvisorAgent até agora.")
        #     else:
        #         st.info("As perguntas sugeridas aparecerão aqui após o upload de dados.")


        # else:
        #     q, ask = None, False

        # Inputs de usuário e pergunta
        if st.session_state["dfs"] is not None:
            st.session_state["user"] = st.text_input("Identificador do usuário", st.session_state["user"])
            q = st.text_input(
                "Ex.: 'Quais variáveis têm outliers?', 'Mostre as distribuições', 'Correlação entre colunas'.",
                key="question_input"
            )

            col_q1, col_q2 = st.columns([4,1])
            with col_q1:
                ask = st.button("Responder")
            with col_q2:
                if st.button("❌"):
                    q = ""
                    st.session_state["question_input"] = ""  # também limpa no campo visível

            # 🔎 Perguntas sugeridas dinamicamente pelo AdvisorAgent
            # st.divider()
            # st.subheader("💡 Perguntas sugeridas pelo AdvisorAgent")

            # from eda_agents.advisor_agent import AdvisorAgent
            # advisor = AdvisorAgent(gemini_api_key=gemini_key)

            # if st.session_state.get("last_answer"):
            #     suggestions_block = advisor.summarize(st.session_state["last_answer"])
            #     content = suggestions_block["content"]

            #     if "Perguntas sugeridas:" in content:
            #         parts = content.split("Perguntas sugeridas:")
            #         perguntas = parts[1].strip().split("\n")

            #         # for p in perguntas:
            #         #     if p.strip() and any(ch.isalpha() for ch in p):
            #         #         if st.button(p.strip(), key=f"suggestion_{p}"):
            #         #             st.session_state["question_input"] = p.strip()
            #         #             st.experimental_rerun()
            #         for p in perguntas:
            #             if p.strip() and any(ch.isalpha() for ch in p):
            #                 if st.button(p.strip(), key=f"suggestion_{hash(p)}"):
            #                     st.session_state["question_input"] = p.strip()
            #                     st.rerun()
            #     else:
            #         st.info("Nenhuma sugestão disponível no momento.")
            # else:
                #st.info("Faça pelo menos uma pergunta para que o AdvisorAgent sugira próximas.")
        else:
            q, ask = None, False
 

    # --- Coluna direita ---
    # with col_right:
    #     if st.session_state["dfs"] is None:
    #         st.warning("📂 Carregue datasets antes de fazer perguntas.")
    #     else:
    #         orch = Orchestrator(st.session_state["dfs"], gemini_api_key=gemini_key)

    #         if ask and q:
    #             with st.spinner("⏳ Processando sua pergunta..."):
    #                 ans = orch.answer(q)
    #             save_qa(st.session_state["user"], q, str(ans))
    #             st.session_state["last_answer"] = ans
    #             st.markdown(f"**Agente chamado:** `{ans['agent']}`")

    #             blocks = ans.get("result", [])
    #             for block in blocks:
    #                 st.subheader(block.get("title", ""))
    #                 btype = block.get("type", "text")
    #                 content = block.get("content")
    #                 if btype == "chart":
    #                     st.plotly_chart(content, use_container_width=True)
    #                 elif btype == "json":
    #                     st.json(content)
    #                 elif btype == "table":
    #                     st.dataframe(content, use_container_width=True)
    #                 else:
    #                     st.write(content)
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
               # st.markdown(f"**Agente chamado:** `{ans['agent']}`")
                # Mostra apenas o agente chamado (sem exibir o objeto completo)
                st.markdown(f"**Agente chamado:** `{ans.get('agent', 'Desconhecido')}`")

                # Verificação: se houver conteúdo mas não blocos (casos raros)
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
                # Placeholder amigável enquanto não há perguntas
                with st.spinner("⏳ Aguardando sua pergunta..."):
                    st.info("Digite sua pergunta no painel à esquerda e clique em **Responder** para começar a análise.")


# --- Aba Resumo Geral ---
with tabs[1]:
    # st.subheader("📊 Resumo Geral e Recomendações")
    # if st.session_state.get("general_summary"):
    #     block = st.session_state["general_summary"]
    #     st.markdown(f"**{block['title']}**")
    #     st.write(block["content"])
    # else:
    #     st.info("Ainda não foi gerado nenhum resumo geral. Faça upload de dados primeiro.")
    if st.session_state.get("general_summary"):
        block = st.session_state["general_summary"]
        content = block["content"]

        # Separa perguntas sugeridas, se existirem
        if "Perguntas sugeridas:" in content:
            parts = content.split("Perguntas sugeridas:")
            resumo = parts[0].strip()
            perguntas = parts[1].strip().split("\n")

            st.write(resumo)

            st.subheader("❓ Perguntas sugeridas")
            for p in perguntas:
                if p.strip() and any(ch.isalpha() for ch in p):  # evita linhas vazias
                    st.markdown(f"- {p.strip()}")
        else:
            st.write(content)
    else:
        st.info("Ainda não foi gerado nenhum resumo geral. Faça upload de dados primeiro.")
        # st.subheader("📊 Resumo Geral e Recomendações")

        # if st.session_state.get("general_summary"):
        #     block = st.session_state["general_summary"]
        #     st.markdown(f"**{block['title']}**")
        #     st.write(block["content"])

        #     # Mostrar tabelas
        #     if "general_summary_tables" in st.session_state:
        #         st.subheader("📋 Dimensões dos datasets")
        #         for tbl in st.session_state["general_summary_tables"]:
        #             st.dataframe(tbl, use_container_width=True)

        #     # Mostrar gráficos
        #     if "general_summary_charts" in st.session_state:
        #         st.subheader("📊 Visualizações rápidas")
        #         for fig in st.session_state["general_summary_charts"]:
        #             st.plotly_chart(fig, use_container_width=True)

        # else:
        #     st.info("Ainda não foi gerado nenhum resumo geral. Faça upload de dados primeiro.")


# --- Aba Histórico ---
with tabs[2]:
    st.subheader("🗂 Histórico completo")

    from datetime import date
    import pandas as pd
    all_users = get_all_users()
    user_options = ["(Todos)"] + all_users

    # 🗓️ Define data atual como padrão
    hoje = date.today()

    col1, col2, col3 = st.columns([1, 1, 1])
    filter_user = col1.selectbox("Usuário", user_options, index=0)
    start_date = col2.date_input("Data inicial", value=hoje)
    end_date = col3.date_input("Data final", value=hoje)

    # Converter para string ISO
    start_str = start_date.isoformat() if start_date else None
    end_str = end_date.isoformat() if end_date else None

    # 🔍 Busca registros
    rows = get_history_filtered(
        user=None if filter_user == "(Todos)" else filter_user,
        start_date=start_str,
        end_date=end_str,
        limit=500  # pega um volume maior para paginar localmente
    )

    if rows:
        df_hist = pd.DataFrame(rows, columns=["Usuário", "Pergunta", "Resposta", "Data"])
        total_registros = len(df_hist)
        itens_por_pagina = 10
        total_paginas = (total_registros - 1) // itens_por_pagina + 1

        # 🧭 Controle de página atual (persistente)
        if "pagina_hist" not in st.session_state:
            st.session_state["pagina_hist"] = 1

        col_a, col_b, col_c = st.columns([1, 2, 1])
        with col_a:
            if st.button("⬅️ Anterior") and st.session_state["pagina_hist"] > 1:
                st.session_state["pagina_hist"] -= 1
        with col_c:
            if st.button("Próxima ➡️") and st.session_state["pagina_hist"] < total_paginas:
                st.session_state["pagina_hist"] += 1

        # 🧮 Intervalo de registros exibidos
        inicio = (st.session_state["pagina_hist"] - 1) * itens_por_pagina
        fim = inicio + itens_por_pagina
        df_pagina = df_hist.iloc[inicio:fim]

        # 📋 Exibir tabela paginada
        st.dataframe(df_pagina, use_container_width=True, height=400)
        st.caption(
            f"📄 Página {st.session_state['pagina_hist']} de {total_paginas}  "
            f"• Mostrando registros de {start_date} a {end_date}  "
            f"• Total de {total_registros} registros."
        )

    else:
        st.info(f"Nenhum histórico encontrado para o período selecionado ({start_date}).")

# with tabs[2]:
#     st.subheader("🗂 Histórico completo")

#     # Obter lista de usuários registados no histórico
#     all_users = get_all_users()
#     user_options = ["(Todos)"] + all_users

#     col1, col2, col3 = st.columns([1, 1, 1])
#     filter_user = col1.selectbox("Usuário", user_options, index=0)
#     start_date = col2.date_input("Data inicial", value=None)
#     end_date = col3.date_input("Data final", value=None)

#     # Converter datas
#     start_str = start_date.isoformat() if start_date else None
#     end_str = end_date.isoformat() if end_date else None

#     rows = get_history_filtered(
#         user=None if filter_user == "(Todos)" else filter_user,
#         start_date=start_str,
#         end_date=end_str,
#         limit=100
#     )

#     if rows:
#         import pandas as pd
#         df_hist = pd.DataFrame(rows, columns=["Usuário", "Pergunta", "Resposta", "Data"])
#         st.dataframe(df_hist, use_container_width=True)
#     else:
#         st.info("Nenhum histórico encontrado para os filtros aplicados.")

# --- Aba Configurações ---
with tabs[3]:
    st.subheader("⚙️ Configurações")

    # Identificador global
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

        # Âmbito do projeto
 # Âmbito / Descrição da solução
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
