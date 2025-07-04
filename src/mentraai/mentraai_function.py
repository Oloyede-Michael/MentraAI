"""
Enhanced MentraAI Empathetic Chatbot with Dataset Integration
Optimized implementation leveraging mental health and health symptom datasets
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from enum import Enum
from dataclasses import dataclass, field
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from contextlib import asynccontextmanager
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

from dotenv import load_dotenv
load_dotenv()

from pydantic import Field, BaseModel, validator
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import trim_messages

from aiq.builder.builder import Builder
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.component_ref import EmbedderRef, FunctionRef, LLMRef
from aiq.data_models.function import FunctionBaseConfig
from aiq.data_models.api_server import AIQChatRequest, AIQChatResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetManager:
    """Efficient dataset management and retrieval system"""
    
    def __init__(self):
        self.datasets = {}
        self.vectorizers = {}
        self.similarity_matrices = {}
        self._initialize_datasets()
    
    def _initialize_datasets(self):
        """Load and preprocess all datasets"""
        try:
            # Load datasets
            self.datasets = {
                'mental_health_dialogue': pd.read_parquet(
                    "hf://datasets/heliosbrahma/mental_health_chatbot_dataset/data/train-00000-of-00001-01391a60ef5c00d9.parquet"
                ),
                'health_symptoms': pd.read_parquet(
                    "hf://datasets/karenwky/pet-health-symptoms-dataset/data/train-00000-of-00001.parquet"
                ),
                'diseases_symptoms': pd.read_csv(
                    "hf://datasets/QuyenAnhDE/Diseases_Symptoms/Diseases_Symptoms.csv"
                ),
                'counseling_conversations': pd.read_json(
                    "hf://datasets/Amod/mental_health_counseling_conversations/combined_dataset.json", 
                    lines=True
                )
            }
            
            logger.info(f"Loaded {len(self.datasets)} datasets successfully")
            
            # Preprocess datasets
            self._preprocess_datasets()
            
            # Create search indices
            self._create_search_indices()
            
        except Exception as e:
            logger.error(f"Failed to initialize datasets: {e}")
            self.datasets = {}
    
    def _preprocess_datasets(self):
        """Clean and standardize datasets"""
        
        # Mental health dialogue preprocessing
        if 'mental_health_dialogue' in self.datasets:
            df = self.datasets['mental_health_dialogue']
            # Standardize column names and clean text
            if 'response' in df.columns:
                df['response'] = df['response'].fillna('').astype(str)
            if 'context' in df.columns:
                df['context'] = df['context'].fillna('').astype(str)
        
        # Counseling conversations preprocessing
        if 'counseling_conversations' in self.datasets:
            df = self.datasets['counseling_conversations']
            # Extract relevant conversation patterns
            if 'conversation' in df.columns:
                df['conversation'] = df['conversation'].fillna('').astype(str)
        
        # Diseases symptoms preprocessing
        if 'diseases_symptoms' in self.datasets:
            df = self.datasets['diseases_symptoms']
            # Clean symptom descriptions
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].fillna('').astype(str)
    
    def _create_search_indices(self):
        """Create TF-IDF indices for efficient similarity search"""
        
        # Mental health dialogue index
        if 'mental_health_dialogue' in self.datasets:
            df = self.datasets['mental_health_dialogue']
            if 'context' in df.columns and not df['context'].empty:
                self.vectorizers['mental_health'] = TfidfVectorizer(
                    max_features=5000,
                    stop_words='english',
                    lowercase=True,
                    ngram_range=(1, 2)
                )
                
                contexts = df['context'].tolist()
                self.similarity_matrices['mental_health'] = self.vectorizers['mental_health'].fit_transform(contexts)
        
        # Counseling conversations index
        if 'counseling_conversations' in self.datasets:
            df = self.datasets['counseling_conversations']
            if 'conversation' in df.columns and not df['conversation'].empty:
                self.vectorizers['counseling'] = TfidfVectorizer(
                    max_features=5000,
                    stop_words='english',
                    lowercase=True,
                    ngram_range=(1, 2)
                )
                
                conversations = df['conversation'].tolist()
                self.similarity_matrices['counseling'] = self.vectorizers['counseling'].fit_transform(conversations)
    
    def find_similar_conversations(self, user_input: str, dataset_type: str = 'mental_health', top_k: int = 3) -> List[Dict]:
        """Find similar conversations using TF-IDF similarity"""
        
        if dataset_type not in self.vectorizers:
            return []
        
        try:
            # Transform user input
            user_vector = self.vectorizers[dataset_type].transform([user_input])
            
            # Calculate similarity
            similarities = cosine_similarity(user_vector, self.similarity_matrices[dataset_type]).flatten()
            
            # Get top k similar conversations
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            results = []
            df = self.datasets[f'{dataset_type}_dialogue' if dataset_type == 'mental_health' else 'counseling_conversations']
            
            for idx in top_indices:
                if similarities[idx] > 0.1:  # Minimum similarity threshold
                    row = df.iloc[idx]
                    results.append({
                        'similarity': float(similarities[idx]),
                        'context': row.get('context', ''),
                        'response': row.get('response', ''),
                        'conversation': row.get('conversation', ''),
                        'index': int(idx)
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Error finding similar conversations: {e}")
            return []
    
    def get_symptom_context(self, user_input: str) -> Dict[str, Any]:
        """Extract symptom-related context from health datasets"""
        
        symptom_context = {
            'potential_symptoms': [],
            'related_conditions': [],
            'severity_indicators': []
        }
        
        user_input_lower = user_input.lower()
        
        # Check diseases_symptoms dataset
        if 'diseases_symptoms' in self.datasets:
            df = self.datasets['diseases_symptoms']
            
            # Look for symptom matches
            for _, row in df.iterrows():
                for col in df.columns:
                    if pd.notna(row[col]):
                        symptom_text = str(row[col]).lower()
                        if any(word in symptom_text for word in user_input_lower.split()):
                            symptom_context['potential_symptoms'].append(symptom_text)
                            if 'disease' in df.columns:
                                symptom_context['related_conditions'].append(str(row.get('disease', '')))
        
        # Check health_symptoms dataset
        if 'health_symptoms' in self.datasets:
            df = self.datasets['health_symptoms']
            
            for _, row in df.iterrows():
                for col in df.columns:
                    if pd.notna(row[col]):
                        symptom_text = str(row[col]).lower()
                        if any(word in symptom_text for word in user_input_lower.split()):
                            symptom_context['potential_symptoms'].append(symptom_text)
        
        # Remove duplicates and limit results
        symptom_context['potential_symptoms'] = list(set(symptom_context['potential_symptoms']))[:5]
        symptom_context['related_conditions'] = list(set(symptom_context['related_conditions']))[:3]
        
        return symptom_context
    
    def get_conversation_patterns(self, emotion_type: str) -> List[str]:
        """Get conversation patterns for specific emotions"""
        
        patterns = []
        
        # Search mental health dialogue for emotion-specific patterns
        if 'mental_health_dialogue' in self.datasets:
            df = self.datasets['mental_health_dialogue']
            
            emotion_keywords = {
                'sadness': ['sad', 'depressed', 'down', 'blue', 'grief', 'sorrow'],
                'anxiety': ['anxious', 'worry', 'nervous', 'panic', 'stress', 'fear'],
                'anger': ['angry', 'mad', 'frustrated', 'rage', 'irritated'],
                'joy': ['happy', 'joy', 'excited', 'glad', 'cheerful'],
                'fear': ['scared', 'afraid', 'frightened', 'terrified', 'phobia']
            }
            
            keywords = emotion_keywords.get(emotion_type, [])
            
            for _, row in df.iterrows():
                context = str(row.get('context', '')).lower()
                response = str(row.get('response', ''))
                
                if any(keyword in context for keyword in keywords):
                    patterns.append(response)
        
        return patterns[:3]  # Return top 3 patterns


class EmotionType(Enum):
    """Enumeration of detectable emotions with clear categorization"""
    # Primary emotions
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    NEUTRAL = "neutral"
    
    # Secondary emotions
    ANXIETY = "anxiety"
    EXCITEMENT = "excitement"
    FRUSTRATION = "frustration"
    CONFIDENCE = "confidence"
    LONELINESS = "loneliness"
    HOPE = "hope"


class ConversationStage(Enum):
    """Conversation flow stages"""
    GREETING = "greeting"
    EXPLORATION = "exploration"
    SUPPORT = "support"
    PROBLEM_SOLVING = "problem_solving"
    CLOSURE = "closure"
    CRISIS = "crisis"


@dataclass
class EmotionAnalysis:
    """Structured emotion analysis results"""
    primary_emotion: EmotionType
    confidence: float = field(default=0.5)
    secondary_emotions: List[EmotionType] = field(default_factory=list)
    emotional_intensity: float = field(default=0.5)
    context_keywords: List[str] = field(default_factory=list)
    suggested_response_tone: str = field(default="supportive")
    
    def __post_init__(self):
        """Validate data ranges"""
        self.confidence = max(0.0, min(1.0, self.confidence))
        self.emotional_intensity = max(0.0, min(1.0, self.emotional_intensity))


@dataclass
class ConversationContext:
    """Enhanced conversation context tracking"""
    user_mood_history: List[EmotionAnalysis] = field(default_factory=list)
    conversation_stage: ConversationStage = field(default=ConversationStage.GREETING)
    topic_thread: str = field(default="general")
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    relationship_building_score: float = field(default=0.0)
    session_start_time: datetime = field(default_factory=datetime.now)
    total_interactions: int = field(default=0)
    similar_conversations: List[Dict] = field(default_factory=list)
    symptom_context: Dict[str, Any] = field(default_factory=dict)


class MentraaiFunctionConfig(FunctionBaseConfig, name="mentraai"):
    """Optimized configuration with better validation"""
    
    # Tool configuration
    tool_names: List[FunctionRef] = Field(
        default_factory=list, 
        description="List of tool references to use"
    )
    
    # LLM model references
    empathy_reasoning_model: LLMRef = Field(
        description="Primary model for empathetic reasoning"
    )
    multi_modal_vision_model: LLMRef = Field(
        description="Model for visual content analysis"
    )
    user_voice_model: LLMRef = Field(
        description="Model for voice input processing"
    )
    agent_voice_output_model: LLMRef = Field(
        description="Model for voice response generation"
    )
    topic_control_model: LLMRef = Field(
        description="Model for conversation flow management"
    )
    
    # Behavioral parameters
    max_history: int = Field(
        default=100, 
        ge=10, 
        le=500, 
        description="Maximum conversation history tokens"
    )
    emotion_sensitivity: float = Field(
        default=0.7, 
        ge=0.1, 
        le=1.0, 
        description="Emotion detection sensitivity"
    )
    empathy_level: str = Field(
        default="high",
        description="Empathy response intensity"
    )

    @validator("empathy_level")
    def validate_empathy_level(cls, v):
        allowed = {"low", "medium", "high"}
        if v not in allowed:
            raise ValueError(f"empathy_level must be one of {allowed}")
        return v
    
    response_style: str = Field(
        default="supportive", 
        description="Response communication style"
    )

    @validator("response_style")
    def validate_response_style(cls, v):
        allowed = {"supportive", "professional", "casual", "therapeutic"}
        if v not in allowed:
            raise ValueError(f"response_style must be one of {allowed}")
        return v
    
    # Document processing
    ingest_glob: str = Field(
        default="**/*.{txt,md,pdf}",
        description="Document ingestion pattern"
    )
    description: str = Field(
        default="Empathic AI Assistant",
        description="Assistant description"
    )
    chunk_size: int = Field(
        default=1024,
        ge=256,
        le=4096,
        description="Document chunk size"
    )
    auto_analyze: bool = Field(
        default=True,
        description="Enable automatic content analysis"
    )
    embedder_name: EmbedderRef = Field(
        description="Embedder model reference"
    )


class LLMManager:
    """Centralized LLM client management with connection pooling"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self._clients: Dict[str, ChatNVIDIA] = {}
        self._base_config = {
            "api_key": api_key,
            "temperature": 0.3,
            "max_tokens": 1024
        }
    
    def get_client(self, model_name: str, **overrides) -> ChatNVIDIA:
        """Get or create LLM client with caching"""
        cache_key = f"{model_name}_{hash(str(sorted(overrides.items())))}"
        
        if cache_key not in self._clients:
            config = {**self._base_config, **overrides}
            self._clients[cache_key] = ChatNVIDIA(
                model=model_name,
                **config
            )
        
        return self._clients[cache_key]


class PromptManager:
    """Centralized prompt management with templates"""
    
    @staticmethod
    def get_emotion_analysis_prompt() -> ChatPromptTemplate:
        """Optimized emotion analysis prompt"""
        return ChatPromptTemplate.from_messages([
            ("system", """
            You are an expert emotion analyst. Analyze the user's message for emotional content.
            
            Provide analysis in this exact JSON format:
            {{
                "primary_emotion": "one of: joy, sadness, anger, fear, surprise, disgust, neutral, anxiety, excitement, frustration, confidence, loneliness, hope",
                "confidence": 0.0-1.0,
                "secondary_emotions": ["emotion1", "emotion2"],
                "emotional_intensity": 0.0-1.0,
                "context_keywords": ["keyword1", "keyword2"],
                "suggested_response_tone": "supportive/encouraging/validating/calming/celebratory"
            }}
            
            Focus on:
            - Explicit emotional words
            - Implicit emotional indicators
            - Context clues
            - Intensity markers
            """),
            ("human", "Analyze: {message}")
        ])
    
    @staticmethod
    def get_empathy_response_prompt() -> ChatPromptTemplate:
        """Enhanced empathetic response generation with dataset context"""
        return ChatPromptTemplate.from_messages([
            ("system", """
            You are an empathetic AI assistant specializing in emotional support and understanding.
            
            Current context:
            - Emotion analysis: {emotion_analysis}
            - Conversation stage: {conversation_stage}
            - Response style: {response_style}
            - User's emotional history: {mood_history}
            
            Dataset insights:
            - Similar conversations: {similar_conversations}
            - Symptom context: {symptom_context}
            - Conversation patterns: {conversation_patterns}
            
            Response guidelines:
            1. **Validate first**: Acknowledge and validate emotions before offering solutions
            2. **Use dataset insights**: Draw from similar conversations and patterns when appropriate
            3. **Match tone**: Align your response with the suggested tone
            4. **Be specific**: Reference specific elements from their message
            5. **Stay present**: Focus on their current experience
            6. **Offer support**: Provide appropriate emotional support
            7. **Maintain boundaries**: Professional but warm
            8. **Health awareness**: Be mindful of potential health-related concerns
            
            Avoid:
            - Minimizing feelings ("at least", "could be worse")
            - Immediate problem-solving without validation
            - Generic responses
            - Overwhelming with questions
            - Medical diagnosis or advice
            """),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])


class EmotionAnalyzer:
    """Dedicated emotion analysis component"""
    
    def __init__(self, llm_manager: LLMManager):
        self.llm_manager = llm_manager
        self.prompt = PromptManager.get_emotion_analysis_prompt()
    
    async def analyze(self, message: str) -> EmotionAnalysis:
        """Analyze emotional content with robust error handling"""
        try:
            client = self.llm_manager.get_client(
                "meta/llama-4-maverick-17b-128e-instruct"
            )
            
            response = await client.ainvoke(
                self.prompt.format_messages(message=message)
            )
            
            return self._parse_emotion_response(str(response.content))
            
        except Exception as e:
            logger.warning(f"Emotion analysis failed: {e}")
            return self._create_fallback_analysis(message)
    
    def _parse_emotion_response(self, content: str) -> EmotionAnalysis:
        """Parse LLM response with fallback handling"""
        try:
            logger.debug(f"Raw emotion response: {content}")
            
            # Extract JSON more robustly
            content = content.strip()
            
            # Remove markdown blocks
            if content.startswith("```"):
                lines = content.split('\n')
                content = '\n'.join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            
            # Find first complete JSON object
            start = content.find('{')
            if start != -1:
                brace_count = 0
                for i, char in enumerate(content[start:], start):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_str = content[start:i+1]
                            data = json.loads(json_str)
                            break
                else:
                    raise json.JSONDecodeError("No complete JSON object found", content, 0)
            else:
                raise json.JSONDecodeError("No JSON object found", content, 0)
            
            return EmotionAnalysis(
                primary_emotion=EmotionType(data["primary_emotion"]),
                confidence=float(data.get("confidence", 0.5)),
                secondary_emotions=[
                    EmotionType(e) for e in data.get("secondary_emotions", [])
                    if e in [emotion.value for emotion in EmotionType]
                ],
                emotional_intensity=float(data.get("emotional_intensity", 0.5)),
                context_keywords=data.get("context_keywords", []),
                suggested_response_tone=data.get("suggested_response_tone", "supportive")
            )
            
        except Exception as e:
            logger.warning(f"JSON parsing failed: {e}")
            return self._create_fallback_analysis(content)
    
    def _create_fallback_analysis(self, message: str) -> EmotionAnalysis:
        """Create basic emotion analysis when parsing fails"""
        message_lower = message.lower()
        
        if any(word in message_lower for word in ["happy", "joy", "great", "excited"]):
            primary_emotion = EmotionType.JOY
        elif any(word in message_lower for word in ["sad", "depressed", "down"]):
            primary_emotion = EmotionType.SADNESS
        elif any(word in message_lower for word in ["angry", "mad", "frustrated"]):
            primary_emotion = EmotionType.ANGER
        elif any(word in message_lower for word in ["worried", "anxious", "nervous"]):
            primary_emotion = EmotionType.ANXIETY
        else:
            primary_emotion = EmotionType.NEUTRAL
        
        return EmotionAnalysis(
            primary_emotion=primary_emotion,
            confidence=0.6,
            suggested_response_tone="supportive"
        )


class EmpathicChatbot:
    """Enhanced empathetic chatbot with dataset integration"""
    
    def __init__(self, config: MentraaiFunctionConfig, builder: Builder):
        self.config = config
        self.builder = builder
        
        # Initialize components
        api_key = os.getenv("NVIDIA_API_KEY")
        if not api_key:
            raise ValueError("NVIDIA_API_KEY environment variable is required")
        
        self.llm_manager = LLMManager(api_key)
        self.emotion_analyzer = EmotionAnalyzer(self.llm_manager)
        self.dataset_manager = DatasetManager()
        self.conversation_context = ConversationContext()
        
        # Initialize prompts
        self.empathy_prompt = PromptManager.get_empathy_response_prompt()
        
        # Initialize tools
        self._initialize_tools()
    
    def _initialize_tools(self):
        """Initialize tools with error handling"""
        self.tools = []
        if self.config.tool_names:
            try:
                self.tools = self.builder.get_tools(
                    tool_names=self.config.tool_names,
                    wrapper_type=LLMFrameworkEnum.LANGCHAIN
                )
                logger.info(f"Initialized {len(self.tools)} tools")
            except Exception as e:
                logger.warning(f"Failed to initialize tools: {e}")
    
    async def process_conversation(self, request: AIQChatRequest) -> AIQChatResponse:
        """Enhanced conversation processing with dataset integration"""
        try:
            messages = request.messages
            if not messages:
                return AIQChatResponse.from_string(
                    "Hello! I'm here to listen and support you. How are you feeling today?"
                )
            
            last_message = messages[-1]
            user_input = self._extract_message_content(last_message)
            
            # Step 1: Analyze emotion
            emotion_analysis = await self.emotion_analyzer.analyze(user_input)
            
            # Step 2: Get dataset insights
            similar_conversations = self.dataset_manager.find_similar_conversations(
                user_input, 'mental_health', top_k=2
            )
            
            symptom_context = self.dataset_manager.get_symptom_context(user_input)
            
            conversation_patterns = self.dataset_manager.get_conversation_patterns(
                emotion_analysis.primary_emotion.value
            )
            
            # Step 3: Update conversation context
            self._update_context(emotion_analysis, similar_conversations, symptom_context)
            
            # Step 4: Prepare chat history
            base_messages = [m for m in messages[:-1] if isinstance(m, BaseMessage)]
            chat_history = self._prepare_chat_history(base_messages)
            
            # Step 5: Generate empathetic response
            response_content = await self._generate_response(
                user_input, chat_history, emotion_analysis, 
                similar_conversations, symptom_context, conversation_patterns
            )
            
            return AIQChatResponse.from_string(response_content)
            
        except Exception as e:
            logger.error(f"Conversation processing failed: {e}")
            return AIQChatResponse.from_string(
                "I'm having some technical difficulties right now, but I want you to know "
                "that I'm here for you. Could you try sharing your thoughts again?"
            )
    
    def _extract_message_content(self, message) -> str:
        """Extract content from message with type handling"""
        if hasattr(message, 'content'):
            content = message.content
            return str(content) if content is not None else ""
        return str(message)
    
    def _update_context(self, emotion_analysis: EmotionAnalysis, 
                       similar_conversations: List[Dict], 
                       symptom_context: Dict[str, Any]):
        """Update conversation context with emotion and dataset insights"""
        self.conversation_context.user_mood_history.append(emotion_analysis)
        self.conversation_context.similar_conversations = similar_conversations
        self.conversation_context.symptom_context = symptom_context
        
        # Keep history manageable
        if len(self.conversation_context.user_mood_history) > 10:
            self.conversation_context.user_mood_history.pop(0)
        
        # Update conversation stage
        self._update_conversation_stage(emotion_analysis)
        
        # Increment interaction counter
        self.conversation_context.total_interactions += 1
    
    def _update_conversation_stage(self, emotion_analysis: EmotionAnalysis):
        """Update conversation stage based on emotion analysis"""
        if emotion_analysis.primary_emotion in [EmotionType.ANGER, EmotionType.FEAR]:
            if emotion_analysis.emotional_intensity > 0.8:
                self.conversation_context.conversation_stage = ConversationStage.CRISIS
            else:
                self.conversation_context.conversation_stage = ConversationStage.SUPPORT
        elif emotion_analysis.primary_emotion == EmotionType.SADNESS:
            self.conversation_context.conversation_stage = ConversationStage.SUPPORT
        elif self.conversation_context.total_interactions == 1:
            self.conversation_context.conversation_stage = ConversationStage.GREETING
        else:
            self.conversation_context.conversation_stage = ConversationStage.EXPLORATION
    
    from typing import Sequence

    def _prepare_chat_history(self, messages: Sequence[BaseMessage]) -> List[BaseMessage]:
        """Prepare and trim chat history"""
        if not messages:
            return []
        
        return trim_messages(
            messages=messages,
            max_tokens=self.config.max_history,
            strategy="last",
            token_counter=len,
            start_on="human",
            include_system=True
        )

    def _generate_fallback_response(self, emotion_analysis: EmotionAnalysis) -> str:
        """Generate a fallback response based on emotion"""
        emotion_responses = {
            EmotionType.SADNESS: "I can sense you're going through something difficult. I'm here to listen.",
            EmotionType.ANGER: "I can hear that you're really frustrated. Those feelings are completely valid.",
            EmotionType.ANXIETY: "I understand you're feeling anxious. Take a deep breath - you're not alone in this.",
            EmotionType.FEAR: "I can feel your worry. It's okay to feel scared - your feelings are important.",
            EmotionType.JOY: "I'm glad to hear some positivity in your message! I'd love to hear more about what's going well.",
            EmotionType.LONELINESS: "I can sense you might be feeling alone. I'm here with you right now.",
            EmotionType.NEUTRAL: "I'm here and ready to listen to whatever you'd like to share."
        }
        return emotion_responses.get(
            emotion_analysis.primary_emotion, 
            "I'm here to support you through whatever you're experiencing."
        )
    
    async def _generate_response(self, 
                               user_input: str, 
                               chat_history: List[BaseMessage],
                               emotion_analysis: EmotionAnalysis,
                               similar_conversations: List[Dict],
                               symptom_context: Dict[str, Any],
                               conversation_patterns: List[str]) -> str:
        """Generate empathetic response with dataset insights"""
        try:
            client = self.llm_manager.get_client(
                "meta/llama-4-maverick-17b-128e-instruct"
            )
            
            # Prepare contexts
            emotion_context = {
                "primary": emotion_analysis.primary_emotion.value,
                "intensity": emotion_analysis.emotional_intensity,
                "confidence": emotion_analysis.confidence,
                "tone": emotion_analysis.suggested_response_tone
            }
            
            mood_history = [
                e.primary_emotion.value 
                for e in self.conversation_context.user_mood_history[-3:]
            ]
            
            # Format similar conversations for context
            similar_conv_summary = []
            for conv in similar_conversations[:2]:  # Use top 2
                if conv.get('response'):
                    similar_conv_summary.append(f"Similar situation response: {conv['response'][:100]}...")
            
            response = await client.ainvoke(
                self.empathy_prompt.format_messages(
                    emotion_analysis=emotion_context,
                    conversation_stage=self.conversation_context.conversation_stage.value,
                    response_style=self.config.response_style,
                    mood_history=mood_history,
                    similar_conversations=similar_conv_summary,
                    symptom_context=symptom_context,
                    conversation_patterns=conversation_patterns[:2],
                    chat_history=chat_history,
                    input=user_input
                )
            )
            
            content = response.content
            return str(content) if content is not None else "I'm here to support you."
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return self._generate_fallback_response(emotion_analysis)
            
    
def _generate_fallback_response(self, emotion_analysis: EmotionAnalysis) -> str:
        """Generate a fallback response based on emotion"""
        emotion_responses = {
            EmotionType.SADNESS: "I can sense you're going through something difficult. I'm here to listen.",
            EmotionType.ANGER: "I can hear that you're really frustrated. Those feelings are completely valid.",
            EmotionType.ANXIETY: "I understand you're feeling anxious. Take a deep breath - you're not alone in this.",
            EmotionType.FEAR: "I can feel your worry. It's okay to feel scared - your feelings are important.",
            EmotionType.JOY: "I'm glad to hear some positivity in your message! I'd love to hear more about what's going well.",
            EmotionType.LONELINESS: "I can sense you might be feeling alone. I'm here with you right now.",
            EmotionType.NEUTRAL: "I'm here and ready to listen to whatever you'd like to share."
        }
        
        return emotion_responses.get(
            emotion_analysis.primary_emotion, 
            "I'm here to support you through whatever you're experiencing."
        )


class ConversationHistoryManager:
    """Enhanced conversation history management with context preservation"""
    
    def __init__(self, max_history: int = 50):
        self.max_history = max_history
        self.conversation_memory: List[Dict[str, Any]] = []
        self.emotion_timeline: List[Tuple[datetime, EmotionAnalysis]] = []
        
    def add_interaction(self, 
                       user_input: str, 
                       bot_response: str, 
                       emotion_analysis: EmotionAnalysis,
                       dataset_insights: Optional[Dict[str, Any]] = None):
        """Add interaction to conversation history"""
        interaction = {
            'timestamp': datetime.now(),
            'user_input': user_input,
            'bot_response': bot_response,
            'emotion_analysis': emotion_analysis,
            'dataset_insights': dataset_insights or {}
        }
        
        self.conversation_memory.append(interaction)
        self.emotion_timeline.append((datetime.now(), emotion_analysis))
        
        # Maintain history limits
        if len(self.conversation_memory) > self.max_history:
            self.conversation_memory.pop(0)
            
        if len(self.emotion_timeline) > self.max_history:
            self.emotion_timeline.pop(0)
    
    def get_emotional_trend(self, last_n: int = 5) -> Dict[str, Any]:
        """Analyze emotional trend over last n interactions"""
        if not self.emotion_timeline:
            return {}
        
        recent_emotions = self.emotion_timeline[-last_n:]
        
        if not recent_emotions:
            return {}
        
        # Calculate trend
        emotions = [e[1].primary_emotion.value for e in recent_emotions]
        intensities = [e[1].emotional_intensity for e in recent_emotions]
        
        trend = {
            'emotions': emotions,
            'avg_intensity': sum(intensities) / len(intensities),
            'intensity_trend': 'increasing' if len(intensities) > 1 and intensities[-1] > intensities[0] else 'stable',
            'dominant_emotion': max(set(emotions), key=emotions.count),
            'emotion_variety': len(set(emotions))
        }
        
        return trend
    
    def get_relevant_context(self, current_emotion: EmotionType, similarity_threshold: float = 0.7) -> List[Dict]:
        """Get relevant historical context based on current emotion"""
        relevant_interactions = []
        
        for interaction in self.conversation_memory[-10:]:  # Look at last 10 interactions
            past_emotion = interaction['emotion_analysis'].primary_emotion
            
            # Check for similar emotional states
            if (past_emotion == current_emotion or 
                past_emotion in [EmotionType.SADNESS, EmotionType.ANXIETY] and 
                current_emotion in [EmotionType.SADNESS, EmotionType.ANXIETY]):
                
                relevant_interactions.append({
                    'user_input': interaction['user_input'],
                    'bot_response': interaction['bot_response'],
                    'emotion': past_emotion.value,
                    'timestamp': interaction['timestamp']
                })
        
        return relevant_interactions[-3:]  # Return last 3 relevant interactions


class CrisisDetectionManager:
    """Enhanced crisis detection with multiple indicators"""
    
    def __init__(self):
        self.crisis_keywords = {
            'suicide': ['kill myself', 'end it all', 'suicide', 'not worth living', 'better off dead'],
            'self_harm': ['cut myself', 'hurt myself', 'self harm', 'self-harm', 'punish myself'],
            'extreme_distress': ['can\'t take it anymore', 'give up', 'hopeless', 'no point', 'can\'t go on'],
            'substance_abuse': ['too much alcohol', 'overdose', 'pills', 'drugs to cope'],
            'violence': ['hurt someone', 'make them pay', 'revenge', 'violent thoughts']
        }
        
        self.crisis_response_templates = {
            'immediate_support': "I'm really concerned about you right now. Your life has value and you matter. Please reach out to a crisis helpline: National Suicide Prevention Lifeline: 988",
            'self_harm': "I'm worried about you wanting to hurt yourself. Please reach out for help: Crisis Text Line: Text HOME to 741741",
            'general_crisis': "I can hear that you're in a lot of pain right now. Please consider reaching out to a mental health professional or crisis support service."
        }
    
    def detect_crisis_indicators(self, message: str, emotion_analysis: EmotionAnalysis) -> Dict[str, Any]:
        """Detect multiple crisis indicators"""
        message_lower = message.lower()
        crisis_detected = False
        crisis_type = None
        crisis_level = 0  # 0-3 scale
        
        # Check for explicit crisis keywords
        for category, keywords in self.crisis_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                crisis_detected = True
                crisis_type = category
                crisis_level = 3 if category in ['suicide', 'self_harm'] else 2
                break
        
        # Check emotional intensity indicators
        if emotion_analysis.emotional_intensity > 0.8:
            if emotion_analysis.primary_emotion in [EmotionType.SADNESS, EmotionType.FEAR, EmotionType.ANGER]:
                crisis_level = max(crisis_level, 1)
                crisis_detected = True
                crisis_type = crisis_type or 'emotional_distress'
        
        # Check for hopelessness indicators
        hopelessness_indicators = ['no hope', 'hopeless', 'pointless', 'meaningless', 'give up']
        if any(indicator in message_lower for indicator in hopelessness_indicators):
            crisis_level = max(crisis_level, 2)
            crisis_detected = True
            crisis_type = crisis_type or 'hopelessness'
        
        return {
            'crisis_detected': crisis_detected,
            'crisis_type': crisis_type,
            'crisis_level': crisis_level,
            'recommended_action': self._get_crisis_action(crisis_type or 'general_crisis', crisis_level)
        }
    
    def _get_crisis_action(self, crisis_type: str, crisis_level: int) -> str:
        """Get recommended crisis intervention action"""
        if crisis_level >= 3:
            return 'immediate_professional_help'
        elif crisis_level >= 2:
            return 'crisis_resources'
        elif crisis_level >= 1:
            return 'enhanced_support'
        else:
            return 'monitor'
    
    def get_crisis_response(self, crisis_info: Dict[str, Any]) -> str:
        """Generate appropriate crisis response"""
        if not crisis_info.get('crisis_detected'):
            return ""
        
        crisis_type = crisis_info.get('crisis_type', 'general_crisis')
        crisis_level = crisis_info.get('crisis_level', 0)
        
        if crisis_level >= 3:
            return self.crisis_response_templates.get('immediate_support', '')
        elif crisis_type == 'self_harm':
            return self.crisis_response_templates.get('self_harm', '')
        else:
            return self.crisis_response_templates.get('general_crisis', '')


class ResponsePersonalizationManager:
    """Advanced response personalization using dataset insights"""
    
    def __init__(self, dataset_manager: DatasetManager):
        self.dataset_manager = dataset_manager
        self.user_preferences = {}
        self.response_history = []
        
    def personalize_response(self, 
                           base_response: str, 
                           user_context: ConversationContext,
                           emotion_analysis: EmotionAnalysis) -> str:
        """Personalize response based on user history and preferences"""
        
        # Get user's preferred response style from history
        preferred_style = self._infer_preferred_style(user_context)
        
        # Adjust response based on emotional trend
        emotional_trend = self._analyze_emotional_pattern(user_context.user_mood_history)
        
        # Get conversation patterns from dataset
        patterns = self.dataset_manager.get_conversation_patterns(
            emotion_analysis.primary_emotion.value
        )
        
        # Apply personalization
        personalized_response = self._apply_personalization_rules(
            base_response, preferred_style, emotional_trend, patterns
        )
        
        return personalized_response
    
    def _infer_preferred_style(self, user_context: ConversationContext) -> str:
        """Infer user's preferred communication style"""
        # Simple heuristic based on conversation history
        if user_context.total_interactions < 3:
            return "formal_supportive"
        
        # Analyze response patterns (this would be more sophisticated in production)
        recent_emotions = user_context.user_mood_history[-5:]
        if not recent_emotions:
            return "balanced"
        
        # If user frequently expresses anxiety, prefer calming responses
        anxiety_count = sum(1 for e in recent_emotions if e.primary_emotion == EmotionType.ANXIETY)
        if anxiety_count > len(recent_emotions) / 2:
            return "calming_focused"
        
        return "balanced"
    
    def _analyze_emotional_pattern(self, mood_history: List[EmotionAnalysis]) -> Dict[str, Any]:
        """Analyze emotional patterns for personalization"""
        if not mood_history:
            return {}
        
        recent_moods = mood_history[-5:]
        
        pattern = {
            'stability': self._calculate_emotional_stability(recent_moods),
            'dominant_emotion': self._get_dominant_emotion(recent_moods),
            'trend': self._get_emotional_trend(recent_moods),
            'intensity_pattern': self._get_intensity_pattern(recent_moods)
        }
        
        return pattern
    
    def _calculate_emotional_stability(self, moods: List[EmotionAnalysis]) -> float:
        """Calculate emotional stability score"""
        if len(moods) < 2:
            return 0.5
        
        # Calculate variance in emotional intensity
        intensities = [mood.emotional_intensity for mood in moods]
        variance = np.var(intensities)
        
        # Convert to stability score (lower variance = higher stability)
        stability = max(0.0, 1.0 - float(variance))
        return float(stability)
    
    def _get_dominant_emotion(self, moods: List[EmotionAnalysis]) -> EmotionType:
        """Get the most frequent emotion"""
        if not moods:
            return EmotionType.NEUTRAL
        
        emotions = [mood.primary_emotion for mood in moods]
        return max(set(emotions), key=emotions.count)
    
    def _get_emotional_trend(self, moods: List[EmotionAnalysis]) -> str:
        """Determine if emotions are improving, worsening, or stable"""
        if len(moods) < 2:
            return "stable"
        
        # Simple trend analysis based on intensity
        intensities = [mood.emotional_intensity for mood in moods]
        
        if intensities[-1] > intensities[0] + 0.1:
            return "intensifying"
        elif intensities[-1] < intensities[0] - 0.1:
            return "improving"
        else:
            return "stable"
    
    def _get_intensity_pattern(self, moods: List[EmotionAnalysis]) -> str:
        """Analyze intensity patterns"""
        if not moods:
            return "unknown"
        
        avg_intensity = sum(mood.emotional_intensity for mood in moods) / len(moods)
        
        if avg_intensity > 0.7:
            return "high_intensity"
        elif avg_intensity < 0.3:
            return "low_intensity"
        else:
            return "moderate_intensity"
    
    def _apply_personalization_rules(self, 
                                   base_response: str, 
                                   preferred_style: str, 
                                   emotional_trend: Dict[str, Any],
                                   patterns: List[str]) -> str:
        """Apply personalization rules to response"""
        
        # This is a simplified version - in production, this would be more sophisticated
        personalized_response = base_response
        
        # Adjust based on preferred style
        if preferred_style == "calming_focused":
            if not any(word in base_response.lower() for word in ["calm", "breathe", "gentle"]):
                personalized_response = f"Take a moment to breathe. {personalized_response}"
        
        # Adjust based on emotional trend
        if emotional_trend.get('trend') == 'improving':
            if not any(word in base_response.lower() for word in ["progress", "better", "positive"]):
                personalized_response += " I'm noticing some positive changes in how you're feeling."
        
        return personalized_response


# Enhanced main function with all components integrated
@register_function(config_type=MentraaiFunctionConfig)
async def mentraai(config: MentraaiFunctionConfig, builder: Builder):
    """
    Enhanced empathetic chatbot with comprehensive dataset integration
    
    Features:
    - Advanced emotion analysis with confidence scoring
    - Dataset-driven response generation
    - Crisis detection and intervention
    - Conversation history management
    - Response personalization
    - Multi-modal support preparation
    """

    # Define the actual callable to be yielded
    async def _mentraai(request: AIQChatRequest) -> AIQChatResponse:
        # Initialize enhanced chatbot
        chatbot = EmpathicChatbot(config, builder)
        
        # Initialize additional managers
        history_manager = ConversationHistoryManager(max_history=config.max_history)
        crisis_manager = CrisisDetectionManager()
        personalization_manager = ResponsePersonalizationManager(chatbot.dataset_manager)
        
        try:
            # Process the conversation
            initial_response = await chatbot.process_conversation(request)
            
            # Extract user input for additional processing
            if request.messages:
                user_input = chatbot._extract_message_content(request.messages[-1])
                
                # Get emotion analysis
                emotion_analysis = await chatbot.emotion_analyzer.analyze(user_input)
                
                # Crisis detection
                crisis_info = crisis_manager.detect_crisis_indicators(user_input, emotion_analysis)
                
                # Handle crisis if detected
                if crisis_info.get('crisis_detected'):
                    crisis_response = crisis_manager.get_crisis_response(crisis_info)
                    if crisis_response:
                        # Prepend crisis response to initial response
                        combined_response = f"{crisis_response}\n\n{str(initial_response)}"
                        return AIQChatResponse.from_string(combined_response)
                
                # Personalize response
                personalized_response = personalization_manager.personalize_response(
                    str(initial_response),
                    chatbot.conversation_context,
                    emotion_analysis
                )
                
                # Update conversation history
                history_manager.add_interaction(
                    user_input=user_input,
                    bot_response=personalized_response,
                    emotion_analysis=emotion_analysis,
                    dataset_insights={
                        'similar_conversations': chatbot.conversation_context.similar_conversations,
                        'symptom_context': chatbot.conversation_context.symptom_context
                    }
                )
                
                return AIQChatResponse.from_string(personalized_response)
            
            return initial_response
            
        except Exception as e:
            logger.error(f"MentraAI processing failed: {e}")
            
            # Fallback response
            fallback_response = (
                "I'm experiencing some technical difficulties, but I want you to know that "
                "I'm here for you. Your feelings are valid and important. If you're in crisis, "
                "please don't hesitate to reach out to a mental health professional or crisis helpline."
            )
            
            return AIQChatResponse.from_string(fallback_response)

    yield _mentraai
