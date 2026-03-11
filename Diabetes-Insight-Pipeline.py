# --------------------------------------------------------------
# Diabetes Insight Pipeline
# --------------------------------------------------------------

import os
import time
import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# ==============================================================
# 1️⃣ CONFIGURAÇÃO
# ==============================================================

load_dotenv()                                   # carrega .env da pasta corrente
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise RuntimeError("⚠️  Variável GROQ_API_KEY não encontrada no .env")

# Cria o cliente Groq
client = Groq(api_key=api_key)                 # <-- SDK oficial da Groq


# ==============================================================
# 2️⃣ PIPELINE
# ==============================================================

class DiabetesInsightPipeline:
    """
    Treina um modelo de regressão logística para predizer diabetes
    e usa a Groq para gerar um insight interpretativo a partir dos
    pesos (feature importance) e da acurácia.
    """

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        # features que vamos usar (pode adaptar)
        self.features = [
            "Glucose",
            "BMI",
            "Age",
            "DiabetesPedigreeFunction",
        ]
        self.results = {}

    # ----------------------------------------------------------
    # 2.1 Treina o modelo e guarda métricas
    # ----------------------------------------------------------
    def run_pipeline(self) -> dict:
        df = pd.read_csv(self.csv_path)

        # separa X (features) e y (rótulo)
        X = df[self.features]
        y = df["Outcome"]

        # split 80/20
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # modelo
        model = LogisticRegression(max_iter=200, solver="lbfgs")
        model.fit(X_train, y_train)

        # métricas
        acc = accuracy_score(y_test, model.predict(X_test))
        weights = dict(zip(self.features, model.coef_[0].tolist()))

        self.results = {
            "accuracy": acc,
            "feature_importance": weights,
        }
        return self.results

    # ----------------------------------------------------------
    # 2.2 Envia prompt para a Groq e obtém a explicação
    # ----------------------------------------------------------
    def ask_ai_insight(self) -> str:
        """
        Envia o dicionário de resultados para a Groq e retorna
        a explicação em linguagem natural.
        """
        # Se o pipeline ainda não foi rodado, aborta
        if not self.results:
            raise RuntimeError(
                "⚠️  run_pipeline() ainda não foi executado. "
                "Execute antes de chamar ask_ai_insight()."
            )

        # Respeita a taxa de requisição da Groq (pelo menos 1 seg entre chamadas)
        print("⏳ Aguardando 2 segundos para respeitar a taxa de requisição...")
        time.sleep(2)

        # Monta o prompt
        prompt = f"""
        Como analista de dados, explique quais fatores aumentam o risco de diabetes
        baseado nestes pesos do modelo: {self.results['feature_importance']}.
        Acurácia do modelo: {self.results['accuracy']:.2%}
        Por favor, escreva em português, usando linguagem clara e, se possível,
        destaque os fatores mais relevantes.
        """

        # ------------------------------------------------------
        # 3️⃣ Chamada à API Groq (chat completion)
        # ------------------------------------------------------
        # Escolha um modelo que você tenha permissão. Alguns exemplos:
        # - "llama3-8b-8192"
        # - "mixtral-8x7b-32768"
        # - "gemma-2b-it"
        model_name = "llama-3.3-70b-versatile"

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "Você é um analista de dados especializado em saúde."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,          # grau de criatividade (0 = determinístico)
                max_tokens=500,           # limite de tokens na resposta
                top_p=1,
                stop=None,
            )
        except Exception as exc:
            raise RuntimeError(f"⚠️  Falha ao chamar a API Groq: {exc}") from exc

        # O conteúdo vem em response.choices[0].message.content
        return response.choices[0].message.content.strip()


# ==============================================================
# 4️⃣ EXECUÇÃO
# ==============================================================

if __name__ == "__main__":
    # Caminho relativo ao script; ajuste se necessário
    CSV_PATH = "diabetes.csv"

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"⚠️  Arquivo CSV não encontrado: {CSV_PATH}")

    pipeline = DiabetesInsightPipeline(CSV_PATH)

    print("\n⚙️  Treinando modelo de classificação...")
    resultados = pipeline.run_pipeline()
    print(f"✅  Acurácia do modelo: {resultados['accuracy']:.2%}")

    print("\n🤖  Gerando insight com a Groq...")
    try:
        insight = pipeline.ask_ai_insight()
        print("\n" + "-" * 50)
        print(insight)
        print("-" * 50 + "\n")
    except Exception as e:
        print(f"⚠️  Erro ao obter insight: {e}")
