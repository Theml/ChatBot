import asyncio
import json
import logging
import os
import random
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from health_data_processor import HealthDataProcessor

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Modelos Pydantic
class ChatMessage(BaseModel):
    message: str
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    timestamp: str
    conversation_id: str
    confidence: Optional[float] = None
    model_info: Optional[str] = None

class HealthTip(BaseModel):
    tip: str
    category: str
    timestamp: str

class ModelStatus(BaseModel):
    ready: bool
    model_name: str
    loading: bool
    error: Optional[str] = None

# FastAPI App
app = FastAPI(
    title="Salus Health Assistant",
    description="Assistente de saúde inteligente com múltiplos modelos LLM",
    version="3.1.0"
)

class SalusHealthBot:
    def __init__(self):
        self.model = None
        self.model_name = "Carregando..."
        self.model_loaded = False
        self.is_loading = True
        self.conversations: Dict[str, List[Dict]] = {}
        self.load_error = None
        
        # # Base de conhecimento melhorada
        self.health_knowledge = self._load_health_knowledge()
        
        # Dicas categorizadas de saúde
        self.health_tips = {
            "nutrição": [
                "Beba pelo menos 8 copos de água por dia para manter-se hidratado",
                "Inclua 5 porções de frutas e vegetais coloridos em sua dieta diária",
                "Consuma proteínas magras como peixes, frango sem pele e leguminosas",
                "Evite alimentos ultraprocessados e ricos em açúcar refinado",
                "Faça refeições regulares a cada 3-4 horas para manter o metabolismo ativo"
            ],
            "exercício": [
                "Pratique pelo menos 150 minutos de atividade física moderada por semana",
                "Inclua exercícios de força muscular 2-3 vezes por semana",
                "Faça alongamentos diários para manter a flexibilidade",
                "Prefira escadas ao invés de elevadores quando possível",
                "Caminhe pelo menos 10.000 passos por dia"
            ],
            "sono": [
                "Durma de 7 a 9 horas por noite para adultos",
                "Mantenha um horário regular para dormir e acordar",
                "Evite telas 1 hora antes de dormir",
                "Mantenha o quarto escuro, silencioso e fresco",
                "Evite cafeína 6 horas antes de dormir"
            ],
            "mental": [
                "Pratique mindfulness ou meditação por 10 minutos diários",
                "Mantenha conexões sociais saudáveis",
                "Pratique gratidão anotando 3 coisas boas do seu dia",
                "Aprenda a dizer 'não' para evitar sobrecarga",
                "Busque ajuda profissional quando necessário"
            ],
            "prevenção": [
                "Lave as mãos frequentemente com água e sabão",
                "Mantenha a vacinação em dia",
                "Faça check-ups médicos regulares",
                "Use protetor solar diariamente",
                "Não fume e evite consumo excessivo de álcool"
            ]
        }
        
        # Instancia o processador de dados de saúde
        self.health_data_processor = HealthDataProcessor('health_knowledge.json')
        
        # Carrega modelo em thread separada
        threading.Thread(target=self._load_available_model, daemon=True).start()
    
    def _load_health_knowledge(self) -> Dict:
        """Carrega base de conhecimento de saúde"""
        return {
            "sintomas_comuns": {
                "dor_cabeca": {
                    "causas": ["estresse", "desidratação", "tensão muscular", "falta de sono"],
                    "remedios": ["descanso", "hidratação", "compressa fria", "relaxamento"],
                    "quando_procurar_medico": ["dor muito intensa", "febre alta", "vômitos", "alterações visuais"]
                },
                "fadiga": {
                    "causas": ["falta de sono", "estresse", "má alimentação", "sedentarismo"],
                    "remedios": ["sono adequado", "exercício regular", "alimentação balanceada", "gerenciamento do estresse"],
                    "quando_procurar_medico": ["fadiga persistente", "perda de peso inexplicada", "outros sintomas associados"]
                }
            },
            "habitos_saudaveis": {
                "alimentacao": {
                    "boas_praticas": ["refeições regulares", "variedade de alimentos", "hidratação adequada"],
                    "evitar": ["excesso de açúcar", "alimentos ultraprocessados", "pular refeições"]
                },
                "exercicio": {
                    "tipos": ["aeróbico", "força", "flexibilidade", "equilíbrio"],
                    "frequencia": "150 min/semana moderado ou 75 min/semana intenso"
                }
            }
        }
    
    def _load_available_model(self):
        """Tenta carregar diferentes modelos LLM disponíveis"""
        models_to_try = [
            self._try_gpt4all,
            self._try_huggingface_local,
            self._fallback_rule_based
        ]
        
        for model_loader in models_to_try:
            try:
                if model_loader():
                    break
            except Exception as e:
                # logger.warning(f"Falha ao carregar modelo: {e}")
                self.load_error = str(e)
                continue
        
        self.is_loading = False
        # logger.info(f"✅ Modelo carregado: {self.model_name}")
    
    def _try_gpt4all(self) -> bool:
        """Tenta carregar GPT4All"""
        try:
            from gpt4all import GPT4All
            
            # Modelos menores e mais confiáveis
            models = [
                "orca-mini-3b-gguf2-q4_0.gguf",
                "DeepSeek-R1-Distill-Qwen-1.5B-Q4_0.gguf",
                "Llama-3.1-MedPalm2-imitate-8B-Instruct.Q4_0.gguf",
            ]
            
            for model_name in models:
                try:
                    self.model = GPT4All(
                        model_name=model_name,
                        allow_download=True,
                        device='cpu'
                    )
                    # Teste mais simples
                    test = self.model.generate("Olá", max_tokens=3)
                    if test:
                        self.model_name = f"GPT4All - {model_name}"
                        self.model_loaded = True
                        return True
                except Exception as e:
                    logger.warning(f"Erro com modelo {model_name}: {e}")
                    continue
            return False
        except ImportError:
            logger.warning("GPT4All não está instalado")
            return False
    
    def _try_huggingface_local(self) -> bool:
        """Tenta carregar modelo do Hugging Face localmente"""
        try:
            from transformers import pipeline
            
            # Modelo muito leve para teste
            model_name = [
                "meditron-7b.Q8_0.gguf",
                "Llama-3.1-MedPalm2-imitate-8B-Instruct.Q4_0.gguf"
            ]
            
            self.model = pipeline(
                "text-generation",
                model=model_name,
                device=-1,  # CPU
                max_new_tokens=100
            )
            # Teste simples
            test = self.model("Olá", max_new_tokens=50, num_return_sequences=1)
            if test:
                self.model_name = f"HuggingFace - {model_name}"
                self.model_loaded = True
                return True
            return False
        except Exception as e:
            logger.warning(f"HuggingFace não disponível: {e}")
            return False
    
    def _fallback_rule_based(self) -> bool:
        """Sistema baseado em regras como fallback"""
        self.model = "rule_based"
        self.model_name = "Sistema Baseado em Regras"
        self.model_loaded = True
        return True
    
    def _generate_with_ollama(self, prompt: str) -> str:
        """Gera resposta usando Ollama"""
        try:
            import requests
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "Llama-3.1-MedPalm2-imitate-8B-Instruct.Q4_0", 
                    "prompt": prompt,
                    "stream": False
                },
                timeout=30
            )
            if response.status_code == 200:
                result = response.json().get("response", "")
                return self._clean_response(result)
            return self._fallback_response()
        except Exception as e:
            logger.error(f"Erro Ollama: {e}")
            return self._fallback_response()
    
    def _generate_with_gpt4all(self, prompt: str) -> str:
        """Gera resposta usando GPT4All com limitação rigorosa"""
        try:
            response = self.model.generate(
                prompt=prompt,
                max_tokens=80,  
                temp=0.3,       
                top_k=20,
                top_p=0.7
            )
            return self._clean_response(response)
        except Exception as e:
            logger.error(f"Erro GPT4All: {e}")
            return self._fallback_response()
    
    def _generate_with_huggingface(self, prompt: str) -> str:
        """Gera resposta usando Hugging Face"""
        try:
            response = self.model(
                prompt,
                max_new_tokens=200,
                num_return_sequences=1,
                temperature=0.9,
                do_sample=True,
                pad_token_id=50256
            )[0]['generated_text']
            return self._clean_response(response.replace(prompt, "").strip())
        except Exception as e:
            logger.error(f"Erro Hugging Face: {e}")
            return self._fallback_response()
    
    def _generate_rule_based(self, user_input: str) -> str:
        """Sistema baseado em regras como fallback"""
        user_input_lower = user_input.lower()
        
        # Tenta identificar intenção na base de conhecimento
        intent_result = self.health_data_processor.identify_intent(user_input)
        if intent_result:
            intent, score = intent_result
            if score > 0.5:
                return self.health_data_processor.get_response(intent)
        
        # Resposta fixa para febre
        if any(word in user_input_lower for word in ["febre", "estou com febre", "muita febre"]):
            return (
                "🌡️ A febre é um sinal de que o corpo está lutando contra uma infecção. "
                "Mantenha-se hidratado, descanse e use antitérmicos apenas se necessário e conforme orientação médica. "
                "Se a febre persistir por mais de 3 dias, for muito alta ou vier acompanhada de outros sintomas graves, procure um médico imediatamente."
            )
        
        # Saudações
        if any(word in user_input_lower for word in ["olá", "oi", "bom dia", "boa tarde", "boa noite"]):
            return "👋 Olá! Sou o Salus, seu assistente de saúde. Como posso ajudar você hoje? Posso dar dicas sobre nutrição, exercícios, sono ou responder dúvidas gerais sobre saúde."
        
        # Detecção de intenções específicas
        if any(word in user_input_lower for word in ["dor", "cabeça", "cefaleia"]):
            return self._get_symptom_response("dor_cabeca")
        
        elif any(word in user_input_lower for word in ["cansado", "fadiga", "exausto", "sono"]):
            return self._get_symptom_response("fadiga")
        
        elif any(word in user_input_lower for word in ["alimentação", "dieta", "comer", "nutrição"]):
            return self._get_nutrition_advice()
        
        elif any(word in user_input_lower for word in ["exercício", "atividade", "treino", "ginástica"]):
            return self._get_exercise_advice()
        
        elif any(word in user_input_lower for word in ["sono", "dormir", "insônia"]):
            return self._get_sleep_advice()
        
        elif any(word in user_input_lower for word in ["estresse", "ansiedade", "mental"]):
            return self._get_mental_health_advice()
        
        else:
            return self._get_general_health_advice()
    
    def _get_symptom_response(self, symptom: str) -> str:
        """Retorna resposta para sintomas específicos"""
        if symptom in self.health_knowledge["sintomas_comuns"]:
            info = self.health_knowledge["sintomas_comuns"][symptom]
            response = f"💡 Para {symptom.replace('_', ' ')}, algumas causas comuns incluem: {', '.join(info['causas'][:3])}.\n\n"
            response += f"Você pode tentar: {', '.join(info['remedios'][:3])}.\n\n"
            response += f"⚠️ Procure um médico se: {', '.join(info['quando_procurar_medico'][:2])}."
            return response
        return self._get_general_health_advice()
    
    def _get_nutrition_advice(self) -> str:
        tip = random.choice(self.health_tips["nutrição"])
        return f"🍎 **Dica de Nutrição:**\n{tip}\n\nLembre-se: uma alimentação equilibrada é fundamental para sua saúde! Que tal incluir mais frutas e vegetais em suas refeições?"
    
    def _get_exercise_advice(self) -> str:
        tip = random.choice(self.health_tips["exercício"])
        return f"💪 **Dica de Exercício:**\n{tip}\n\nO exercício regular fortalece seu corpo e mente! Comece devagar e aumente gradualmente a intensidade."
    
    def _get_sleep_advice(self) -> str:
        tip = random.choice(self.health_tips["sono"])
        return f"😴 **Dica de Sono:**\n{tip}\n\nUm bom sono é essencial para sua recuperação e bem-estar! Tente criar uma rotina relaxante antes de dormir."
    
    def _get_mental_health_advice(self) -> str:
        tip = random.choice(self.health_tips["mental"])
        return f"🧠 **Dica de Saúde Mental:**\n{tip}\n\nCuidar da mente é tão importante quanto cuidar do corpo. Lembre-se: buscar ajuda é sinal de força!"
    
    def _get_general_health_advice(self) -> str:
        categories = list(self.health_tips.keys())
        category = random.choice(categories)
        tip = random.choice(self.health_tips[category])
        return f"💚 **Dica de Saúde ({category.title()}):**\n{tip}\n\nCuidar da sua saúde é um investimento no seu futuro! Tem alguma área específica em que posso ajudar?"
    
    def _fallback_response(self) -> str:
        """Resposta de fallback mais segura"""
        responses = [
            "🤖 Posso ajudar com dicas gerais de saúde. O que gostaria de saber?",
            "💡 Que tal uma dica sobre nutrição ou exercícios?",
            "🏥 Para orientações específicas, recomendo consultar um profissional de saúde"
        ]
        return random.choice(responses)
    
    def _clean_response(self, response: str) -> str:
        """Limpa e formata a resposta de forma mais agressiva"""
        if not response:
            return self._fallback_response()
        
        response = response.strip()
        
        # Remove prefixos comuns
        prefixes = ["RESPOSTA:", "Salus:", "Bot:", "A:", "R:", "Human:", "Assistant:"]
        for prefix in prefixes:
            if response.upper().startswith(prefix.upper()):
                response = response[len(prefix):].strip()
        
        # Limita tamanho mais rigorosamente
        if len(response) > 150:
            sentences = response.split('. ')
            response = '. '.join(sentences[:2])  # Máximo de 2 frases
            if not response.endswith('.'):
                response += '... [resumo automático]'
        
        # Filtra respostas perigosas
        blacklist = ["diagnóstico", "medicamento", "tratamento", "receita", "prescrevo"]
        if any(word in response.lower() for word in blacklist):
            return "⚠️ Recomendo consultar um profissional de saúde para orientações específicas."
        
        # Adiciona emoji se necessário
        if not any(emoji in response for emoji in ['💡', '🌟', '💚', '🏥', '⚕️']):
            response = f"💡 {response}"
        
        return response
    
    def get_health_prompt(self, user_message: str, context: str = "") -> str:
        """Cria prompt otimizado para saúde com contexto"""
        return f"""
Você é Salus, um assistente virtual especializado em saúde. Sua função é fornecer informações gerais sobre bem-estar.

REGRAS ABSOLUTAS:
1. Mantenha respostas curtas (máximo 3 frases)
2. Se não souber a resposta, diga "Não tenho informações sobre isso"

CONTEXTO DISPONÍVEL:
{context}

PERGUNTA: {user_message}

RESPOSTA:"""
    
    async def generate_response(self, user_message: str, conversation_id: Optional[str] = None) -> tuple[str, float, str]:
        """Gera resposta usando o modelo disponível"""
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
        # Garante que conversation_id nunca será None
        if not isinstance(conversation_id, str):
            conversation_id = str(conversation_id)
        
        try:
            # Obter contexto relevante
            context = self.health_data_processor.get_context_for_gpt(user_message)

            # Garante que a conversa existe antes de salvar qualquer mensagem
            if conversation_id not in self.conversations:
                self.conversations[conversation_id] = []

            # Salva mensagem do usuário
            self.conversations[conversation_id].append({
                "role": "user",
                "content": user_message,
                "timestamp": datetime.now().isoformat()
            })

            # Gera resposta baseada no modelo disponível
            if "GPT4All" in self.model_name:
                prompt = self.get_health_prompt(user_message, context)
                response = self._generate_with_gpt4all(prompt)
                confidence = 0.8
            elif "HuggingFace" in self.model_name:
                prompt = self.get_health_prompt(user_message, context)
                response = self._generate_with_huggingface(prompt)
                confidence = 0.7
            else:  # Rule-based
                response = self._generate_rule_based(user_message)
                confidence = 0.9

            # Salva resposta no histórico
            self.conversations[conversation_id].append({
                "role": "assistant",
                "content": response,
                "timestamp": datetime.now().isoformat()
            })

            # Mantém apenas últimas 20 mensagens
            if len(self.conversations[conversation_id]) > 20:
                self.conversations[conversation_id] = self.conversations[conversation_id][-20:]

            return response, confidence, conversation_id
        
        except Exception as e:
            logger.error(f"Erro ao gerar resposta: {e}")
            return self._fallback_response(), 0.3, conversation_id
    
    def get_random_tip(self, category: Optional[str] = None) -> Dict:
        """Retorna dica aleatória de saúde"""
        if category and category in self.health_tips:
            tip = random.choice(self.health_tips[category])
        else:
            category = random.choice(list(self.health_tips.keys()))
            tip = random.choice(self.health_tips[category])
        
        return {
            "tip": tip,
            "category": category,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_model_status(self) -> Dict:
        """Retorna status do modelo"""
        return {
            "ready": self.model_loaded,
            "model_name": self.model_name,
            "loading": self.is_loading,
            "error": self.load_error
        }

# Instância global
salus_bot = SalusHealthBot()

# Configuração de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# Configuração de templates
templates = Jinja2Templates(directory="templates")

# Rotas da API
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Página principal"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(chat_message: ChatMessage):
    """Endpoint principal do chat"""
    try:
        message = chat_message.message.strip()
        
        if not message:
            raise HTTPException(status_code=400, detail="Mensagem não pode estar vazia")
        
        if len(message) > 1000:
            raise HTTPException(status_code=400, detail="Mensagem muito longa")
        
        response, confidence, conversation_id = await salus_bot.generate_response(
            message, 
            chat_message.conversation_id
        )
        
        return ChatResponse(
            response=response,
            timestamp=datetime.now().isoformat(),
            conversation_id=conversation_id,
            confidence=confidence,
            model_info=salus_bot.model_name
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro no chat: {e}")
        raise HTTPException(status_code=500, detail="Erro interno do servidor")

@app.get("/api/model/status", response_model=ModelStatus)
async def model_status():
    """Status do modelo - endpoint correto para o frontend"""
    status = salus_bot.get_model_status()
    return ModelStatus(**status)

@app.get("/api/health/tip")
async def get_health_tip(category: Optional[str] = None):
    """Dica de saúde aleatória - endpoint correto para o frontend"""
    return salus_bot.get_random_tip(category)

@app.get("/api/health")
async def health_check():
    """Health check geral"""
    return {
        "status": "healthy" if salus_bot.model_loaded else ("loading" if salus_bot.is_loading else "degraded"),
        "model_loaded": salus_bot.model_loaded,
        "is_loading": salus_bot.is_loading,
        "timestamp": datetime.now().isoformat(),
        "model_info": salus_bot.get_model_status()
    }

@app.get("/api/categories")
async def get_categories():
    """Lista categorias de dicas disponíveis"""
    return {"categories": list(salus_bot.health_tips.keys())}

@app.get("/api/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Recupera histórico de conversa"""
    history = salus_bot.conversations.get(conversation_id, [])
    return {
        "conversation_id": conversation_id,
        "messages": history,
        "total_messages": len(history)
    }

@app.get("/test")
async def test_endpoint():
    """Endpoint de teste"""
    return {
        "message": "Salus Health Assistant está funcionando!",
        "model": salus_bot.model_name,
        "status": "loaded" if salus_bot.model_loaded else ("loading" if salus_bot.is_loading else "error"),
        "timestamp": datetime.now().isoformat()
    }

# Função principal
def main():
    """Executa o servidor"""
    logger.info("🚀 Iniciando Salus Health Assistant v3.1...")
    logger.info("📚 Documentação: http://localhost:8000/docs")
    logger.info("🌐 Interface: http://localhost:8000")
    logger.info("🧪 Teste: http://localhost:8000/test")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )

if __name__ == "__main__":
    main()