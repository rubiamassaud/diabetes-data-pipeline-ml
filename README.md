# Pipeline de Predição de Diabetes: ETL & Machine Learning

Este projeto desenvolve uma pipeline completa de processamento de dados e análise preditiva para o diagnóstico de diabetes. O foco principal é demonstrar boas práticas de **Engenharia de Dados** e a aplicação de modelos de **Machine Learning** supervisionado.

## 🛠️ Tecnologias e Ferramentas
* **Linguagem:** Python
* **Manipulação de Dados:** Pandas, Numpy
* **Machine Learning:** Scikit-Learn (KNN, Logistic Regression)
* **Gestão de Ambiente:** Virtualenv & Git

## 📈 Pipeline de Dados (Fluxo de Trabalho)
1. **Extração:** Leitura de dados clínicos via CSV.
2. **Transformação (ETL):** Tratamento de valores inconsistentes e normalização de features.
3. **Modelagem:** Treinamento comparativo entre algoritmos de classificação.
4. **Avaliação:** Comparação de acurácia entre modelos para suporte à decisão.

## 🚀 Como executar
1. Clone o repositório: `git clone https://github.com/seu-usuario/diabetes-data-pipeline-ml.git`
2. Crie um ambiente virtual: `python -m venv venv`
3. Instale as dependências: `pip install -r requirements.txt`
4. Execute o script: `python previsao-diabetes.py`

## 📊 Resultados
O projeto compara a performance dos modelos KNN e Regressão Logística, permitindo identificar qual algoritmo apresenta melhor generalização para este conjunto de dados.
