# 🩺 Diabetes Insight Pipeline

Pipeline que treina um modelo de classificação de diabetes e gera interpretações em linguagem natural usando IA generativa (Gemini).

---

## 💡 Como funciona

1. Carrega um dataset de diabetes em CSV
2. Treina um modelo de **Regressão Logística** com as features mais relevantes
3. Envia os pesos aprendidos para o **Gemini 2.0 Flash**
4. Retorna uma análise interpretativa dos fatores de risco em linguagem humana

---

## 🗂️ Estrutura do projeto

```
diabetes-insight-pipeline/
│
├── diabetes_insight_pipeline.py  # Pipeline principal
├── diabetes.csv                  # Dataset de entrada
├── .env                          # Variáveis de ambiente (não versionar)
├── requirements.txt              # Dependências do projeto
└── README.md
```

---

## ⚙️ Instalação

**1. Clone o repositório**
```bash
git clone https://github.com/rubiamassaud/diabetes-insight-pipeline.git
cd diabetes-insight-pipeline
```

**2. Crie e ative um ambiente virtual**
```bash
python -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate         # Windows
```

**3. Instale as dependências**
```bash
pip install -r requirements.txt
```

**4. Configure a variável de ambiente**

Crie um arquivo `.env` na raiz do projeto:
```env
GEMINI_API_KEY=sua_chave_aqui
```
> Obtenha sua chave em [Google AI Studio](https://aistudio.google.com/app/apikey)

---

## ▶️ Uso

```bash
python diabetes_insight_pipeline.py
```

**Saída esperada:**
```
⚙️ Treinando modelo...
✅ Acurácia: 77.92%

🤖 Gerando Insight com IA...
⏳ Aguardando 10 segundos para respeitar a cota...
------------------------------
[Análise do Gemini sobre os fatores de risco]
------------------------------
```

---

## 📊 Features utilizadas no modelo

| Feature | Descrição |
|---|---|
| `Glucose` | Concentração de glicose no plasma |
| `BMI` | Índice de massa corporal |
| `Age` | Idade do paciente |
| `DiabetesPedigreeFunction` | Histórico familiar de diabetes |

---

## 📦 Dependências

```
pandas
scikit-learn
python-dotenv
google-genai
```

---

## 📁 Dataset

O projeto utiliza o [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database), disponível no Kaggle.

---

## 🤖 Tecnologias

- **Python 3.10+**
- **scikit-learn** — treinamento do modelo
- **Pandas** — manipulação dos dados
- **Gemini 2.0 Flash** — geração de insights com IA
- **python-dotenv** — gerenciamento de variáveis de ambiente
