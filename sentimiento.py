import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import torch
from textblob import TextBlob
nltk.download('vader_lexicon')
from transformers import BertTokenizer, BertForSequenceClassification

from googletrans import Translator
texto_es = "tengo un mal sentimiento"
translator = Translator()
texto_en = translator.translate(texto_es, src="es", dest="en").text

def sentimient_transformers(frase):
   model_name = "bert-base-uncased"
   tokenizer = BertTokenizer.from_pretrained(model_name)
   model = BertForSequenceClassification.from_pretrained(model_name)
   inputs = tokenizer(frase, return_tensors="pt")
   outputs = model(**inputs)

   probs = torch.softmax(outputs.logits, dim=1)
   probs_list = probs.tolist()

   probabilidad_positiva = probs_list[0][0]
   probabilidad_negativa = probs_list[0][1]
   print("\nResultado del análisis de sentimiento para la frase:")
   print(frase)

   return f"\nProbabilidad con transformers de clase positiva: {probabilidad_positiva:.2f}% \nProbabilidad con transformers de clase negativa: {probabilidad_negativa:.2f}%"

print(sentimient_transformers(texto_en))

def sentimient_TextBlob(frase):
    tb = TextBlob(frase)
    polaridad = tb.sentiment.polarity
    return f'\nLa Polaridad de la frase con TextBlob es: {polaridad:.2f}'

print(sentimient_TextBlob(texto_en))


def sentimient_sia(frase):
   sia = SentimentIntensityAnalyzer()

   # Realizar el análisis de sentimiento
   sentimiento = sia.polarity_scores(frase)

   print("\nPuntuación de polaridad con SIA:", sentimiento['compound'])

   if sentimiento['compound'] >= 0.05:
      print("Sentimiento: Positivo")
   elif sentimiento['compound'] <= -0.05:
      print("Sentimiento: Negativo")
   else:
      print("Sentimiento: Neutral")

sentimient_sia(texto_en)