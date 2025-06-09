import os
import joblib
import google.generativeai as genai
import pandas as pd
from dotenv import load_dotenv
from googletrans import Translator
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

class HelperFunctions:
    @staticmethod
    def Average(numList, rateList):
        total = 0
        for it, r in zip(numList, rateList):
            total += it * r
        return total / sum(rateList) 

    @staticmethod
    def RedefineLabel(percentage):
        return "詐騙 Scam" if percentage >= 50 else "普通 Normal"

    @staticmethod
    def get_prediction_proba(classifier, question_tfidf):
        prediction_proba = classifier.predict_proba(question_tfidf)[0]
        spam_proba = prediction_proba[1] * 100
        ham_proba = prediction_proba[0] * 100
        return spam_proba, ham_proba

    @staticmethod
    def AskingQuestion(question):
        try:
            response = chat.send_message(question).text
            return response
        except Exception as e:
            print(f"An error occured in AskingQuestion: {e}")

    @staticmethod
    async def Translate(translator: Translator, message, source_language='auto', target_language='en'):
        translated = await translator.translate(message, src=source_language, dest=target_language)
        return translated.text

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash-exp")
chat = model.start_chat(history=[])

model_lists = [
    ("sklearn (邏輯迴歸 Logistic Regression)", "LRclassifier", 1),
    ("sklearn (支援向量機 Support Vector Classification)", "SVCclassifier", 1),
    ("sklearn (單純貝氏 Naive Bayes)", "NBclassifier", 1),
    ("sklearn (隨機梯度下降 Stochastic Gradient Descent)", "SGDclassifier", 1),
    ("sklearn (決策樹 Decision Tree)", "DTclassifier", 1),
    ("sklearn (隨機森林 Random Forest)", "RFclassifier", 1),
    ("sklearn (堆疊 Stacking)", "STACKclassifier", 1)
]

class Classifiers:
    def __init__(self):
        pass

    @staticmethod
    def __train_models__() -> tuple:
        """
        Train all the sklearn models.

        Parameters:
            None

        Return:
            classifiers, Xtfidf, Ytfidf, vectorizer
        """
        
        # Models and configuration
        vectorizer = TfidfVectorizer()
        LRclassifier = LogisticRegression(n_jobs=-1)
        SVCclassifier = CalibratedClassifierCV(LinearSVC(dual=False), n_jobs=-1)
        NBclassifier = MultinomialNB(alpha=0.08451, fit_prior=True)
        SGDclassifier = CalibratedClassifierCV(SGDClassifier(n_jobs=-1, loss='hinge'))
        DTclassifier = DecisionTreeClassifier()
        RFclassifier = RandomForestClassifier(n_jobs=-1)
        STACKclassifier = StackingClassifier(estimators=[
            ('lr', LRclassifier), 
            ('svc', SVCclassifier), 
            ('nb', NBclassifier), 
            ('sgd', SGDclassifier),
            ('dt', DTclassifier),
            ('rf', RFclassifier)
        ], final_estimator=LogisticRegression(), n_jobs=-1)

        try:
            labels, messages = [], []
            file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.txt")
            with open(file_path, encoding="utf-8") as f:
                dataList = f.read().split('\n')
                for i, line in enumerate(dataList):
                    if not line.strip():
                        continue

                    try:
                        label, message = line.split('\t')
                        labels.append(1 if label == 'spam' else 0)
                        messages.append(message)
                    except ValueError as e:
                        pass
                        continue

            X_train, X_test, y_train, y_test = train_test_split(messages, labels, test_size=0.2)
            X_train_tfidf = vectorizer.fit_transform(X_train)
            X_test_tfidf = vectorizer.transform(X_test)

            classifiers = {
                "LRclassifier": LRclassifier,
                "SVCclassifier": SVCclassifier,
                "NBclassifier": NBclassifier,
                "SGDclassifier": SGDclassifier,
                "STACKclassifier": STACKclassifier,
                "DTclassifier": DTclassifier,
                "RFclassifier": RFclassifier
            }

            for name, classifier in classifiers.items():
                classifier.fit(X_train_tfidf, y_train)
                print(f"{name} is trained.")

            return classifiers, X_test_tfidf, y_test, vectorizer
        except Exception as e:
            return None, None, None, None

    @staticmethod
    def save_data_and_models():
        """
        Save training data and trained models.

        Parameters:
            regenerate_models: bool

        Return:
            None        
        """

        models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
            print(f"Created models directory: {models_dir}")
        
        vectorizer_path = os.path.join(models_dir, "vectorizer.pkl")
        expected_classifier_pkls = [os.path.join(models_dir, f"{classifierKey}.pkl") for _, classifierKey, _ in model_lists]
        all_requirements_path = [vectorizer_path] + expected_classifier_pkls

        existance_check = all(os.path.exists(path) for path in all_requirements_path)

        if not existance_check:
            # regenerate classfiers and vectorizer
                classifiers, Xtfidf, Ytfidf, vectorizer = Classifiers.__train_models__()
                joblib.dump(vectorizer, "./models/vectorizer.pkl")
                for name, model_obj in classifiers.items():
                    model_file_path = os.path.join(models_dir, f"{name}.pkl")
                    joblib.dump(model_obj, model_file_path)

                accuracy_data = []
                if Xtfidf is None or Ytfidf is None:
                    print("Xtfidf or Ytfidf is not available from training. Skipping accuracy calculation.")
                else:
                    for model_name, classifierKey, _ in model_lists:
                        if classifierKey in classifiers:
                            test_classifier = classifiers[classifierKey]
                            y_pred = test_classifier.predict(Xtfidf)

                            accuracy = accuracy_score(Ytfidf, y_pred) * 100
                            recall = recall_score(Ytfidf, y_pred) * 100
                            precision = precision_score(Ytfidf, y_pred) * 100
                            accuracy_data.append({
                                "模型 Model": model_name,
                                "準確度 Accuracy": f"{accuracy:.2f}%",
                                "召回率 Recall": f"{recall:.2f}%",
                                "精準度 Precision": f"{precision:.2f}%"
                            })
                        else:
                            print(f"Classifier {classifierKey} not found in trained models. Skipping for accuracy calculation.")

                if accuracy_data:
                    accuracy_df = pd.DataFrame(accuracy_data)
                    accuracy_csv_path = os.path.join(models_dir, "accuracy.csv")
                    accuracy_df.to_csv(accuracy_csv_path, index=False)
                    print(f"Accuracy data saved to {accuracy_csv_path}")
                    print("Updated accuracy data:\n", accuracy_df.to_string())
                else:
                    print("No accuracy data to save (possibly due to missing classifiers or no models trained).")
                        
                    print(f"All classifier models and vectorizer saved to {models_dir}")      

class MainFunction:
    def __init__(self):
        pass

    @staticmethod
    async def get_label(
            message: str
        ):
        
        """
        Check whether a message is spam or not.

        Parameters:
            message: str

        Return:
            a dictionary includes: {
                "模型 Model": "加權平均分析結果 Weighted Average Analysis Result", 
                "結果 Result": "Error - No models processed", 
                "加權倍率 Rate": 0, 
                "詐騙訊息機率 Scam Probability": "N/A", 
                "普通訊息機率 Normal Probability": "N/A"
            }
        """
        
        try:
            # Translation and AI Judgement
            translator = Translator()
            translation = await HelperFunctions.Translate(translator, message)
            
            AiJudgement = HelperFunctions.AskingQuestion(f"""How much percentage do you think this message is a spamming message (only consider this message, not considering other environmental variation)? 
                Answer in this format: "N" where N is a float between 0-100 (13.62, 85.72, 50.60, 5.67, 100.00, 0.00 etc.)
                message: {translation}""")

            AiJudgePercentage = float(AiJudgement)
            AiJudgePercentageRate = 1

            # Model Analysis
            vectorizer = joblib.load("./models/vectorizer.pkl")
            question_tfidf = vectorizer.transform([translation])
            results_data = []

            # Load .pkl models from ./models/ directory
            classifiers = {}
            # Construct path relative to the current file to find the "models" directory
            models_load_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

            if os.path.exists(models_load_dir) and os.path.isdir(models_load_dir):
                for pkl_filename in os.listdir(models_load_dir):
                    if pkl_filename.endswith(".pkl"):
                        classifier_name_key = pkl_filename[:-4]  # Remove .pkl extension
                        model_file_path = os.path.join(models_load_dir, pkl_filename)
                        try:
                            classifiers[classifier_name_key] = joblib.load(model_file_path)
                            # print(f"Successfully loaded model: {classifier_name_key} from {model_file_path}") # Optional: for debugging
                        except Exception as e:
                            print(f"Error loading model {model_file_path}: {e}")
                            pass
            else:
                print(f"Models directory not found: {models_load_dir}. No sklearn models will be loaded for prediction.")

            for model_name, classifierKey, rate in model_lists:
                working_classifier = classifiers[classifierKey]
                spam_proba, ham_proba = HelperFunctions.get_prediction_proba(working_classifier, question_tfidf)
                results_data.append({
                    "模型 Model": model_name,
                    "結果 Result": HelperFunctions.RedefineLabel(spam_proba),
                    "加權倍率 Rate": rate,
                    "詐騙訊息機率 Scam Probability": f"{spam_proba:.2f}%",
                    "普通訊息機率 Normal Probability": f"{ham_proba:.2f}%"
                })

            # Add Gemini results
            results_data.append({
                "模型 Model": "Google Gemini",
                "結果 Result": HelperFunctions.RedefineLabel(AiJudgePercentage),
                "加權倍率 Rate": AiJudgePercentageRate,
                "詐騙訊息機率 Scam Probability": f"{AiJudgePercentage:.2f}%",
                "普通訊息機率 Normal Probability": f"{100.0 - AiJudgePercentage:.2f}%"
            })

            # Calculate final result
            valid_results = [d for d in results_data if d["結果 Result"] not in ["Error", "Not Loaded"] and d["詐騙訊息機率 Scam Probability"] != "N/A"]
            if not valid_results:
                print("No valid model results to calculate final average.")
                # Return a default or error indicator for result_row
                return {"模型 Model": "加權平均分析結果 Weighted Average Analysis Result", 
                        "結果 Result": "Error - No models processed", 
                        "加權倍率 Rate": 0, 
                        "詐騙訊息機率 Scam Probability": "N/A", 
                        "普通訊息機率 Normal Probability": "N/A"
                        }

            spam_percentages = [float(d["詐騙訊息機率 Scam Probability"].rstrip('%')) for d in valid_results]
            rates = [d["加權倍率 Rate"] for d in valid_results]
            final_spam_percentage = HelperFunctions.Average(spam_percentages, rates)
            final_ham_percentage = 100.0 - final_spam_percentage

            # Add final result
            result = {
                "模型 Model": "加權平均分析結果 Weighted Average Analysis Result",
                "結果 Result": HelperFunctions.RedefineLabel(final_spam_percentage),
                "加權倍率 Rate": sum(rates),
                "詐騙訊息機率 Scam Probability": f"{final_spam_percentage:.2f}%",
                "普通訊息機率 Normal Probability": f"{final_ham_percentage:.2f}%"
            }
            return result
        except Exception as e:
            print(f"An error occured in get_label: {e}")

    def get_training_data():
        models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        accuracy_csv_path = os.path.join(models_dir, "accuracy.csv")
        if os.path.exists(accuracy_csv_path):
            try:
                accuracy_df = pd.read_csv(accuracy_csv_path)
                print("Existing accuracy data:\n", accuracy_df.to_string())
            except Exception as e:
                print(f"Could not read existing accuracy.csv: {e}") 
    
if __name__ == "__main__":
    Classifiers.save_data_and_models()