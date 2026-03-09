import pandas as pd
import time
import os
from google import genai
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# --- CONFIGURAÇÃO ---
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


class DiabetesInsightPipeline:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.features = ['Glucose', 'BMI', 'Age', 'DiabetesPedigreeFunction']
        self.results = {}

    def run_pipeline(self):
        df = pd.read_csv(self.csv_path)

        X = df[self.features]
        y = df['Outcome']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = LogisticRegression(max_iter=200)
        model.fit(X_train, y_train)

        acc = accuracy_score(y_test, model.predict(X_test))

        weights = dict(zip(self.features, model.coef_[0].tolist()))

        self.results = {
            "accuracy": acc,
            "feature_importance": weights
        }

        return self.results

    def ask_ai_insight(self):

        print("⏳ Aguardando 10 segundos para respeitar a cota...")
        time.sleep(10)

        prompt = f"""
        Como analista de dados, explique quais fatores aumentam o risco de diabetes
        baseado nestes pesos do modelo: {self.results['feature_importance']}.
        Acurácia do modelo: {self.results['accuracy']:.2%}
        """

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )

        return response.candidates[0].content.parts[0].text


if __name__ == "__main__":

    pipeline = DiabetesInsightPipeline('diabetes.csv')

    print("⚙️ Treinando modelo...")
    res = pipeline.run_pipeline()

    print(f"✅ Acurácia: {res['accuracy']:.2%}")

    print("\n🤖 Gerando Insight com IA...")

    try:
        resultado = pipeline.ask_ai_insight()

        print("-" * 30)
        print(resultado)
        print("-" * 30)

    except Exception as e:
        print(f"\n⚠️ Erro persistente na API: {e}")
