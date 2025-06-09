# Salus Health Assistant

Salus é um assistente virtual de saúde desenvolvido em Python, utilizando FastAPI e modelos de linguagem natural (LLM) locais. Ele responde dúvidas sobre bem-estar, nutrição, exercícios, sono e hábitos saudáveis, sempre recomendando a consulta com profissionais de saúde para casos específicos.

## Funcionalidades
- Respostas automáticas sobre saúde, nutrição, exercícios, sono e prevenção
- Base de conhecimento customizável (health_knowledge.json)
- Sistema híbrido: utiliza regras, base de conhecimento e modelos LLM locais (GPT4All, HuggingFace)
- Histórico de conversas por sessão
- Dicas de saúde aleatórias
- Interface web moderna (FastAPI + HTML/CSS)

## Como executar
1. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
2. Execute o servidor:
   ```bash
   python salus2.py
   ```
3. Acesse a interface em [http://localhost:8000](http://localhost:8000)

## Estrutura dos arquivos
- `salus2.py`: Backend principal (FastAPI, lógica do bot)
- `health_data_processor.py`: Processamento da base de conhecimento
- `health_knowledge.json`: Base de conhecimento em saúde
- `templates/index.html`: Interface web
- `static/index.css`: Estilos da interface

## Personalização
- Adicione ou edite perguntas/respostas em `health_knowledge.json` para expandir o conhecimento do bot.
- As dicas de saúde podem ser alteradas diretamente no código ou integradas à base de conhecimento.

## Observações
- O Salus não faz diagnósticos, não prescreve medicamentos e não substitui um profissional de saúde.
- Para dúvidas críticas, sempre procure um médico.

## Principais dependências

- **fastapi**: Framework web moderno e rápido para construção da API REST do chatbot.
- **uvicorn**: Servidor ASGI leve e eficiente, usado para rodar a aplicação FastAPI.
- **jinja2**: Motor de templates utilizado para renderizar páginas HTML dinâmicas.
- **python-multipart**: Suporte a formulários e uploads de arquivos na API.
- **gpt4all**: Biblioteca para rodar modelos de linguagem LLM locais (como orca-mini, Llama, etc).
- **pydantic**: Validação e tipagem de dados para as requisições e respostas da API.
- **requests**: (opcional, usado internamente) Para chamadas HTTP a modelos externos como Ollama.
- **transformers**: (opcional, usado internamente) Para rodar modelos HuggingFace localmente.

Cada biblioteca tem um papel fundamental para garantir a interface web, a API, a integração com modelos de linguagem e a validação dos dados.

---
Desenvolvido para fins educacionais e de apoio à promoção da saúde.
