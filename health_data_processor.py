import json
import random
from typing import Dict, List, Any, Optional, Tuple
from difflib import SequenceMatcher

class HealthDataProcessor:
    def __init__(self, data_path: str):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self._build_intent_index()

    def _build_intent_index(self):
        """Cria índice rápido para intenções"""
        self.intent_index = {}
        for category in self.data['categories'].values():
            for topic in category['topics']:
                for pattern in topic['patterns']:
                    self.intent_index[pattern.lower()] = topic

    def identify_intent(self, user_input: str) -> Optional[Tuple[str, float]]:
        """Identifica intenção com correspondência aproximada"""
        user_input = user_input.lower()
        best_score = 0
        best_intent = None
        
        for pattern, topic in self.intent_index.items():
            if pattern in user_input:
                score = len(pattern) / len(user_input)
                if score > best_score:
                    best_score = score
                    best_intent = topic['intent']
        
        # fallback: busca por similaridade se não encontrou por substring
        if not best_intent:
            for category in self.data['categories'].values():
                for topic in category['topics']:
                    for pattern in topic['patterns']:
                        score = SequenceMatcher(None, user_input, pattern.lower()).ratio()
                        if score > best_score and score > 0.6:
                            best_score = score
                            best_intent = topic['intent']
        
        return (best_intent, best_score) if best_intent else None

    def get_response(self, intent: str) -> str:
        """Retorna resposta para uma intenção"""
        for category in self.data['categories'].values():
            for topic in category['topics']:
                if topic['intent'] == intent:
                    return random.choice(topic['responses'])
        return "Desculpe, não entendi. Poderia reformular?"

    def _format_response(self, item: Dict[str, Any]) -> str:
        """Formata a resposta com informações contextuais relevantes"""
        response = item['responses'][0]  # Podemos melhorar para alternar respostas
        
        context = item.get('contexto', {})
        if refs := context.get('referencias'):
            response += f"\n\nFonte(s): {', '.join(refs)}"
        
        if context.get('gravidade') == 'alta':
            response = "⚠️ ATENÇÃO: " + response

        return response

    def get_emergency_contacts(self) -> List[str]:
        """Retorna contatos de emergência disponíveis"""
        emergency = self.data['categories'].get('emergencia', {}).get('sinais_alerta', [])
        for item in emergency:
            if context := item.get('contexto', {}):
                if contacts := context.get('contatos_emergencia'):
                    return contacts
        return []

    def get_metadata(self) -> Dict[str, str]:
        """Retorna os metadados da base de conhecimento"""
        return self.data.get('metadata', {})

    def find_best_match(self, user_input: str) -> Optional[Dict]:
        best_score = 0
        best_match = None

        for category in self.data['categories'].values():
            for topic in category.get('topics', []):
                for pattern in topic['patterns']:
                    score = SequenceMatcher(None, user_input.lower(), pattern.lower()).ratio()
                    if score > best_score and score > 0.6:  # Limiar de similaridade
                        best_score = score
                        best_match = topic

        return best_match

    def get_context_for_gpt(self, user_input: str) -> str:
        """Retorna contexto formatado para melhorar respostas"""
        match = self.find_best_match(user_input)
        if match:
            return (
                "CONTEXTO RELEVANTE:\n"
                f"Tópico: {match['intent']}\n"
                f"Palavras-chave: {', '.join(match['keywords'])}\n"
                f"Orientação: {match['responses'][0]}"
            )
        return "Nenhum contexto relevante encontrado."
