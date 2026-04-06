Diabetes Insight Pipeline 🚀
Um pequeno pipeline que treina um modelo de regressão logística para predizer a presença de diabetes (dataset Pima Indians) e, em seguida, usa a API da Groq para gerar um insight interpretativo em linguagem natural a partir dos pesos (feature importance) e da acurácia do modelo.

Objetivo – Demonstrar como combinar Machine Learning tradicional (scikit‑learn) com LLMs (Groq) para criar relatórios automatizados e de fácil compreensão.

🎯 Visão geral
Carrega o dataset
diabetes.csv (colunas padrão do Pima Indians Diabetes Database).
Treina um modelo de LogisticRegression usando as features:
Glucose, BMI, Age e DiabetesPedigreeFunction.
Calcula a acurácia e extrai os coeficientes (importância das variáveis).
Envia essas informações para a API da Groq (modelo llama-3.3-70b-versatile por padrão).
Recebe um texto em português que explica quais fatores mais influenciam o risco de diabetes e a qualidade do modelo.
Tudo isso está encapsulado na classe
DiabetesInsightPipeline (arquivopipeline.py).

🛠️ Requisitos
Ferramenta	Versão mínima
Python	3.9
pip	21+
Git (opcional)	—
Bibliotecas Python
pandas
scikit-learn
python-dotenv
groq

Nota:
groq é o SDK oficial da Groq. Ele será instalado via
pip (ou uv/poetry se preferir).

⚙️ Instalação
# 1️⃣ Clone o repositório (ou copie os arquivos)
git clone https://github.com/SEU_USUARIO/diabetes-insight-pipeline.git
cd diabetes-insight-pipeline

# 2️⃣ Crie e ative um ambiente virtual (recomendado)
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate      # Windows

# 3️⃣ Instale as dependências
pip install -r requirements.txt

Alternativas – Se usar
uv ou poetry, basta adaptar o comando de instalação (ex.:uv pip install -r requirements.txt).

Arquivo
requirements.txt

pandas
scikit-learn
python-dotenv
groq

🔐 Configuração da API Groq
Crie uma conta em https://groq.com/ e obtenha sua API key.
Na raiz do projeto, crie o arquivo
.env

com o seguinte conteúdo:
# .env
GROQ_API_KEY=sk-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX


▶️ Como usar
# Certifique‑se de que o .env está configurado e o ambiente virtual ativo
python pipeline.py

O script:

Verifica a presença do diabetes.csv.
Treina o modelo e imprime a acurácia.
Aguarda 2 s (respeitando a taxa de requisição da Groq).
Envia o prompt para a Groq e exibe o insight.
Personalizando
Dataset – Troque o caminho da variável
CSV_PATH

no final do arquivo ou passe como argumento (modifique o if __name__ == "__main__" conforme desejar).
Features – Edite a lista
self.features

na classe para incluir/excluir variáveis.
Modelo LLM – Altere
model_name

para outro modelo suportado (ex.: llama3-8b-8192, mixtral-8x7b-32768).

📄 Saída esperada
⚙️  Treinando modelo de classificação...
✅  Acurácia do modelo: 78.33%

🤖  Gerando insight com a Groq...

--------------------------------------------------
Como analista de dados, explique quais fatores aumentam o risco de diabetes...
[texto gerado pela Groq em português]

Fatores mais relevantes:
- **Glucose**: alto nível de glicemia está fortemente associado ao risco.
- **BMI**: índice de massa corporal acima da média eleva a probabilidade.
- **Age**: risco aumenta com a idade.
- **DiabetesPedigreeFunction**: indica histórico familiar e também tem peso significativo.

A acurácia do modelo (78,33%) indica que ele captura bem a relação entre essas variáveis e o diagnóstico, embora ainda haja margem para melhorias (ex.: incluir outras features ou usar modelos mais complexos).
--------------------------------------------------

O texto será diferente a cada execução (devido à temperatura > 0), mas seguirá a mesma estrutura de explicação.

📁 Estrutura de pastas
diabetes-insight-pipeline/
│
├─ .env                 # <-- sua API key (não versionado)
├─ .gitignore           # inclui .env, __pycache__, .venv, etc.
├─ requirements.txt
├─ diabetes.csv         # dataset Pima Indians (inclua ou baixe)
├─ pipeline.py          # script principal (código que você enviou)
└─ README.md            # <-- este arquivo


