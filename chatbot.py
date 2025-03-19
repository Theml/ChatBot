import random
import nltk
from nltk.chat.util import Chat, reflections

<!-- Antes de tentar executar rodar os comandos `python -m venv venv`,`.\venv\Scripts\Activate.ps1`, `pip install nltk`, `python -m downloader all` --!>
pares = [
    [
        r"Oi|Olá|E ai",
        ["Olá", "Como posso ajudar você?", "Oi como está?"],
    ],
    [
        r"Qual o seu nome?",
        ["Meu nome é Chatbot", "Você pode me chamar de Chatbot", "Sou o Chatbot"],
    ],
    [
        r"(.*)\?",
        ["Desculpe, não tenho uma resposta para essa pergunta.", "Pode reformular a pergunta?"],
    ],
]

pares.extend([
    [r"(.*)", ["Entendi. Diga-me mais.", "Pode me contar mais sobre isso?", "Interessante. Conte me mais..."]],
])

reflections = {
    "eu" : "você",
    "meu" : "seu",
    "você" : "eu",
    "seu" : "meu",
    "eu sou": "você é",
    "você é" : "eu sou",
    "você estava" : "eu estava",
    "eu estava" : "você estava",
}

chatbot = Chat(pares, reflections)

def iniciar_chat():
    while True:
        entrada = input("Você: ")
        if entrada.lower() == "sair":
            print("Chatbot: Adeus!")
            break
        response = chatbot.respond(entrada)
        print("Chatbot:", response)
