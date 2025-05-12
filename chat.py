from gpt4all import GPT4All
from pathlib import Path

# Caminho para o modelo
model_path = Path(r"C:\Users\Luisg\Documents\Faculdade\IA\Chatbot\models" )

model_name="Mistral-Small-3.1-24B-Instruct-2503.gguf"

# Inicializa o modelo GPT4All
llm = GPT4All(
    model_name=model_name,
    model_path=str(model_path),
    allow_download=False
)

# Testa o modelo com uma sessão de chat
with llm.chat_session() as session:
    resposta = session.generate("How can I run LLMs efficiently on my laptop?", max_tokens=1024)
    print(resposta)

# Base do prompt
prompt_base = "Você é um assistente de saúde e de hábitos saudáveis."

print("Chatbot em português incluído! Digite 'sair' para encerrar.\n")

while True:
    pergunta = input("Usuário: ")
    if pergunta.lower() == "sair":
        break

    # Gera a resposta com o modelo LLM
    prompt = f"{prompt_base} Usuário: {pergunta}\nBot:"
    resposta = llm.generate(prompt=prompt, max_tokens=1024)
    print("Bot:", resposta)