import os, asyncio
import google.generativeai as genai
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

class MainFunctions:
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
            result = chat.send_message(question)
        except Exception as e:
            pass
        return result.text
    
    @staticmethod
    async def Translate(translator: Translator, message, source_language='auto', target_language='en'):
        translated = await translator.translate(message, src=source_language, dest=target_language)
        return translated.text

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash-exp")
chat = model.start_chat(history=[])

models = [
    ("sklearn (邏輯迴歸 Logistic Regression)", "LRclassifier", 1),
    ("sklearn (支援向量機 Support Vector Classification)", "SVCclassifier", 1),
    ("sklearn (單純貝氏 Naive Bayes)", "NBclassifier", 1),
    ("sklearn (隨機梯度下降 Stochastic Gradient Descent)", "SGDclassifier", 1),
    ("sklearn (決策樹 Decision Tree)", "DTclassifier", 1),
    ("sklearn (隨機森林 Random Forest)", "RFclassifier", 1),
    ("sklearn (堆疊 Stacking)", "STACKclassifier", 1)
]

def train_models() -> tuple:
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

        return classifiers, X_test_tfidf, y_test, vectorizer
    except Exception as e:
        return None, None, None, None

def get_label(
        message: str, 
        models: list, 
        classifiers: dict, 
        Xtfidf: list, 
        Ytfidf: list, 
        vectorizer: TfidfVectorizer, 
        translator: Translator
    ):
    
    """
    Check whether a message is spam or not.

    Parameters:
        message: str
        models: list
        classifiers: dict
        Xtfidf: list
        Ytfidf: list
        vectorizer: TfidfVectorizer
        translator: Translator

    Return:
        result: str
    """
    accuracy_data = []
    for model_name, classifierKey, _ in models:
        test_classifier = classifiers[classifierKey]
        y_pred = test_classifier.predict(Xtfidf)

        accuracy = accuracy_score(Ytfidf, y_pred) * 100
        recall = recall_score(Ytfidf, y_pred) * 100
        precision = precision_score(Ytfidf, y_pred) * 100
        accuracy_data.append({
            "模型 Model": model_name, 
            "準確度 Accuracy": accuracy,
            "召回率 Recall": recall,
            "精準度 Precision": precision
        })
    
    # Translation and AI Judgement
    translation = asyncio.run(MainFunctions.Translate(translator, message))
    
    AiJudgement = MainFunctions.AskingQuestion(f"""How much percentage do you think this message is a spamming message (only consider this message, not considering other environmental variation)? 
        Answer in this format: "N" where N is a float between 0-100 (13.62, 85.72, 50.60, 5.67, 100.00, 0.00 etc.)
        message: {translation}""")

    AiJudgePercentage = float(AiJudgement)
    AiJudgePercentageRate = 1

    # Model Analysis
    question_tfidf = vectorizer.transform([translation])
    results_data, result_row = [], []

    for model_name, classifierKey, rate in models:
        working_classifier = classifiers[classifierKey]
        spam_proba, ham_proba = MainFunctions.get_prediction_proba(working_classifier, question_tfidf)
        results_data.append({
            "模型 Model": model_name,
            "結果 Result": MainFunctions.RedefineLabel(spam_proba),
            "加權倍率 Rate": rate,
            "詐騙訊息機率 Scam Probability": f"{spam_proba:.2f}%",
            "普通訊息機率 Normal Probability": f"{ham_proba:.2f}%"
        })

    # Add Gemini results
    results_data.append({
        "模型 Model": "Google Gemini",
        "結果 Result": MainFunctions.RedefineLabel(AiJudgePercentage),
        "加權倍率 Rate": AiJudgePercentageRate,
        "詐騙訊息機率 Scam Probability": f"{AiJudgePercentage:.2f}%",
        "普通訊息機率 Normal Probability": f"{100.0 - AiJudgePercentage:.2f}%"
    })

    # Calculate final result
    spam_percentages = [float(d["詐騙訊息機率 Scam Probability"].rstrip('%')) for d in results_data]
    rates = [d["加權倍率 Rate"] for d in results_data]
    final_spam_percentage = MainFunctions.Average(spam_percentages, rates)
    final_ham_percentage = 100.0 - final_spam_percentage

    # Add final result
    result_row.append({
        "模型 Model": "加權平均分析結果 Weighted Average Analysis Result",
        "結果 Result": MainFunctions.RedefineLabel(final_spam_percentage),
        "加權倍率 Rate": sum(rates),
        "詐騙訊息機率 Scam Probability": f"{final_spam_percentage:.2f}%",
        "普通訊息機率 Normal Probability": f"{final_ham_percentage:.2f}%"
    })

    return MainFunctions.RedefineLabel(final_spam_percentage)