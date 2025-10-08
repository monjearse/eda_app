# 🤖 EDA Baseada em Agentes Inteligentes (LangChain + Gemini + Streamlit)

## 🧠 Visão Geral
Esta aplicação implementa uma **Análise Exploratória de Dados (EDA)** assistida por **agentes de IA especializados**.  
Cada agente é responsável por uma tarefa específica da exploração de dados (estatísticas, gráficos, correlações, anomalias e recomendações), todos coordenados por um **Orchestrator** que utiliza **LLM (Gemini)** para compreender a intenção do utilizador e acionar o agente adequado.

---

## 🧩 Arquitetura
- **Streamlit** → Interface interativa para upload, perguntas e exibição de resultados.
- **LangChain + Gemini** → Processamento de linguagem natural e interpretação semântica das perguntas.
- **Agentes Inteligentes** → Módulos especializados:  
  - `AnalystAgent` → Estatísticas descritivas.  
  - `VisualizerAgent` → Gráficos interativos (Plotly).  
  - `PatternAgent` → Correlações e padrões.  
  - `AnomalyAgent` → Detecção de outliers (IQR).  
  - `AdvisorAgent` → Resumos, recomendações e conclusões.  
- **SQLite** → Armazenamento do histórico de interações (perguntas e respostas).

---

## ⚙️ Instalação e Configuração

### 1️⃣ Clonar o repositório
```bash
git clone https://github.com/teu_usuario/eda-agentes.git
cd eda-agentes
```

### 2️⃣ Instalar dependências
```bash
pip install -r requirements.txt
```

### 3️⃣ Configurar variáveis de ambiente

#### Localmente (`.env`)
```bash
GEMINI_API_KEY="AIxxxx"
GEMINI_MODEL="gemini-2.0-flash-exp"
SQLITE_DB="relatorios_nf.db"
LANG="pt"
```

#### No Streamlit Cloud (`st.secrets` - formato TOML)
```toml
GEMINI_API_KEY = "AIxxxx"
GEMINI_MODEL = "gemini-2.0-flash-exp"
SQLITE_DB = "relatorios_nf.db"
LANG = "pt"
```

---

## 🚀 Execução Local
```bash
streamlit run app_eda.py
```
A aplicação será aberta em: [http://localhost:8501](http://localhost:8501)

---

## ☁️ Deploy no Streamlit Cloud

1. Fazer **push** do projeto para o GitHub.  
2. No [Streamlit Cloud](https://share.streamlit.io), criar uma nova app:  
   - **Main file path:** `app_eda.py`  
   - **Branch:** `main`  
3. Adicionar as chaves em **Settings → Secrets** (ver formato TOML acima).  
4. Clicar em **Deploy**.

---

## 🗂 Estrutura do Projeto

```
eda-agentes/
│
├── app_eda.py               # Interface principal Streamlit
├── requirements.txt
├── .env                     # Configuração local (ignorado pelo Git)
├── eda_agents/
│   ├── orchestrator.py
│   ├── analyst_agent.py
│   ├── visualizer_agent.py
│   ├── pattern_agent.py
│   ├── anomaly_agent.py
│   └── advisor_agent.py
│
├── memory/
│   ├── memory.py
│
└── utils_eda.py
```

---

## 📊 Funcionalidades Principais

| Função | Descrição |
|--------|------------|
| **Upload de CSVs** | Carregamento simultâneo de múltiplos ficheiros. |
| **Perguntas em linguagem natural** | O utilizador pergunta e os agentes respondem com gráficos e textos. |
| **Geração automática de gráficos** | Histogramas, boxplots, barras e pizza. |
| **Interpretação dos resultados** | LLM gera explicações em tom humano. |
| **Histórico diário** | Consultas armazenadas em SQLite, filtradas por data. |
| **Paginação** | 10 registros por página. |
| **Resumo geral** | AdvisorAgent gera conclusões e recomendações. |

---

## 🧠 Exemplo de Fluxo
```text
Usuário: "Quais variáveis têm outliers?"
 → Orchestrator → AnomalyAgent → cálculo IQR + gráficos + explicação
Usuário: "Resuma as descobertas anteriores."
 → Orchestrator → AdvisorAgent → resumo + recomendações + perguntas sugeridas
```

---

## 💾 Banco de Dados
Armazenamento local via SQLite:
```sql
CREATE TABLE historico (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    usuario TEXT,
    pergunta TEXT,
    resposta TEXT,
    data TEXT DEFAULT CURRENT_TIMESTAMP
);
```

---

## 💡 Fallback Inteligente
Cada agente usa o LLM como prioridade, mas tem um **modo local** (fallback) que garante resposta mesmo sem acesso à API.  
Exemplo:
- Se o LLM falhar → gera resumo local com cálculos estatísticos.
- Se o Streamlit estiver offline → mantém histórico e gráficos locais.

---

## 🧭 Licença
Projeto desenvolvido por **Arsénio António Monjane (I2A2 - Institut d'Intelligence Artificielle Appliquée)**  
Distribuído sob licença **MIT**.

---

## 📘 Documentação Técnica
Para mais detalhes sobre os agentes e o fluxo de execução, consulte o ficheiro:  
📄 [`ARQUITETURA_SOLUCAO.md`](./ARQUITETURA_SOLUCAO.md)
