"""
Optimized MentraAI Empathetic Chatbot Implementation
Clean, efficient, and well-structured code with comprehensive error handling
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Union, Tuple
from enum import Enum
from dataclasses import dataclass, field
import os
import json
from datetime import datetime
from contextlib import asynccontextmanager

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
        """Optimized empathetic response generation"""
        return ChatPromptTemplate.from_messages([
            ("system", """
            You are an empathetic AI assistant specializing in emotional support and understanding.
            
            Current context:
            - Emotion analysis: {emotion_analysis}
            - Conversation stage: {conversation_stage}
            - Response style: {response_style}
            - User's emotional history: {mood_history}
            
            Response guidelines:
            1. **Validate first**: Acknowledge and validate emotions before offering solutions
            2. **Match tone**: Align your response with the suggested tone
            3. **Be specific**: Reference specific elements from their message
            4. **Stay present**: Focus on their current experience
            5. **Offer support**: Provide appropriate emotional support
            6. **Maintain boundaries**: Professional but warm
            
            Avoid:
            - Minimizing feelings ("at least", "could be worse")
            - Immediate problem-solving without validation
            - Generic responses
            - Overwhelming with questions
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
            # Clean up response - remove code blocks if present
            # content = content.strip()
            # if content.startswith("```json"):
            #     content = content[7:]
            # if content.endswith("```"):
            #     content = content[:-3]
            
            # data = json.loads(content.strip())
            #log for debugging
            logger.debug(f"Raw response: {content}")
            
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
        # except (json.JSONDecodeError, KeyError, ValueError) as e:
        #     logger.warning(f"Failed to parse emotion response: {e}")
        #     return self._create_fallback_analysis(content)
    
    def _create_fallback_analysis(self, message: str) -> EmotionAnalysis:
        """Create basic emotion analysis when parsing fails"""
        # Simple keyword-based fallback
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
    """Main empathetic chatbot implementation"""
    
    def __init__(self, config: MentraaiFunctionConfig, builder: Builder):
        self.config = config
        self.builder = builder
        
        # Initialize components
        api_key = os.getenv("NVIDIA_API_KEY")
        if not api_key:
            raise ValueError("NVIDIA_API_KEY environment variable is required")
        
        self.llm_manager = LLMManager(api_key)
        self.emotion_analyzer = EmotionAnalyzer(self.llm_manager)
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
        """Main conversation processing pipeline"""
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
            
            # Step 2: Update conversation context
            self._update_context(emotion_analysis)
            
            # Step 3: Prepare chat history
            base_messages = [m for m in messages[:-1] if isinstance(m, BaseMessage)]
            chat_history = self._prepare_chat_history(base_messages)
            
            # Step 4: Generate empathetic response
            response_content = await self._generate_response(
                user_input, chat_history, emotion_analysis
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
    
    def _update_context(self, emotion_analysis: EmotionAnalysis):
        """Update conversation context with new emotion data"""
        self.conversation_context.user_mood_history.append(emotion_analysis)
        
        # Keep history manageable
        if len(self.conversation_context.user_mood_history) > 10:
            self.conversation_context.user_mood_history.pop(0)
        
        # Update conversation stage based on emotion
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
    
    async def _generate_response(self, 
                               user_input: str, 
                               chat_history: List[BaseMessage],
                               emotion_analysis: EmotionAnalysis) -> str:
        """Generate empathetic response"""
        try:
            client = self.llm_manager.get_client(
                "meta/llama-4-maverick-17b-128e-instruct"
            )
            
            # Prepare context
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
            
            response = await client.ainvoke(
                self.empathy_prompt.format_messages(
                    emotion_analysis=emotion_context,
                    conversation_stage=self.conversation_context.conversation_stage.value,
                    response_style=self.config.response_style,
                    mood_history=mood_history,
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
            EmotionType.ANXIETY: "It sounds like you're feeling worried about something. That must be really stressful.",
            EmotionType.JOY: "I can feel your positive energy! It's wonderful to hear from you.",
            EmotionType.FEAR: "I can sense your concern, and I want you to know that you're not alone with this."
        }
        
        return emotion_responses.get(
            emotion_analysis.primary_emotion,
            "I'm here to listen and support you. Would you like to tell me more about what's on your mind?"
        )


@register_function(config_type=MentraaiFunctionConfig)
async def mentraai_function(config: MentraaiFunctionConfig, builder: Builder):
    """Optimized main function entry point"""
    
    logger.info("Initializing MentraAI Empathetic Chatbot")
    
    try:
        # Initialize chatbot
        chatbot = EmpathicChatbot(config, builder)
        
        # Define response function
        async def response_handler(input_message: AIQChatRequest) -> AIQChatResponse:
            """Handle incoming chat requests"""
            return await chatbot.process_conversation(input_message)
        
        # Yield function info
        yield FunctionInfo.create(single_fn=response_handler)
        
    except GeneratorExit:
        logger.info("MentraAI function generator exited")
    except Exception as e:
        logger.error(f"MentraAI function initialization failed: {e}")
        raise
    finally:
        logger.info("MentraAI function cleanup completed")