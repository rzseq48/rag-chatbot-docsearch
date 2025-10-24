"""
LLM provider for RAG chatbot
"""
import os
import logging
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    def generate_response(self, prompt: str, context: Optional[str] = None) -> str:
        """Generate response from LLM"""
        pass
    
    @abstractmethod
    def generate_response_with_context(self, query: str, context_documents: List[str]) -> str:
        """Generate response using retrieved context"""
        pass

class MockLLMProvider(LLMProvider):
    """
    Mock LLM provider for testing
    In production, replace with actual LLM (OpenAI, Anthropic, etc.)
    """
    
    def __init__(self):
        logger.info("Initialized MockLLMProvider")
    
    def generate_response(self, prompt: str, context: Optional[str] = None) -> str:
        """Generate mock response"""
        if context:
            return f"Based on the context provided, here's a response to: {prompt}\n\nContext: {context[:100]}..."
        return f"Mock response to: {prompt}"
    
    def generate_response_with_context(self, query: str, context_documents: List[str]) -> str:
        """Generate response using context documents"""
        context_summary = " ".join([doc[:50] + "..." for doc in context_documents[:3]])
        return f"Based on the retrieved documents, here's what I found about '{query}':\n\n{context_summary}"

class OpenAILLMProvider(LLMProvider):
    """
    OpenAI LLM provider
    Requires OPENAI_API_KEY environment variable
    """
    
    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.model = model
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        try:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("openai package is required for OpenAILLMProvider")
        
        logger.info(f"Initialized OpenAILLMProvider with model {model}")
    
    def generate_response(self, prompt: str, context: Optional[str] = None) -> str:
        """Generate response using OpenAI API"""
        try:
            messages = []
            if context:
                messages.append({
                    "role": "system",
                    "content": f"Use the following context to answer the question: {context}"
                })
            
            messages.append({
                "role": "user",
                "content": prompt
            })
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=1000,
                temperature=0.7
            )
            
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise
    
    def generate_response_with_context(self, query: str, context_documents: List[str]) -> str:
        """Generate response using context documents"""
        context = "\n\n".join(context_documents)
        return self.generate_response(query, context)

class AnthropicLLMProvider(LLMProvider):
    """
    Anthropic Claude LLM provider
    Requires ANTHROPIC_API_KEY environment variable
    """
    
    def __init__(self, model: str = "claude-3-sonnet-20240229"):
        self.model = model
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is required")
        
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError("anthropic package is required for AnthropicLLMProvider")
        
        logger.info(f"Initialized AnthropicLLMProvider with model {model}")
    
    def generate_response(self, prompt: str, context: Optional[str] = None) -> str:
        """Generate response using Anthropic API"""
        try:
            system_prompt = ""
            if context:
                system_prompt = f"Use the following context to answer the question: {context}"
            
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return response.content[0].text
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise
    
    def generate_response_with_context(self, query: str, context_documents: List[str]) -> str:
        """Generate response using context documents"""
        context = "\n\n".join(context_documents)
        return self.generate_response(query, context)

class GroqLLMProvider(LLMProvider):
    """
    Groq LLM provider for ultra-fast inference
    
    Groq provides 10-100x faster inference than traditional cloud APIs by using
    specialized hardware optimized for large language models.
    
    Features:
    - Ultra-fast inference (sub-second response times)
    - Multiple model options (Llama, Mixtral, Gemma)
    - Cost-effective pricing
    - Global edge locations for low latency
    
    Available Models:
    - llama3-8b-8192: Fast, efficient, good for most tasks
    - llama3-70b-8192: More capable, better reasoning
    - mixtral-8x7b-32768: Excellent for complex reasoning
    - gemma2-9b-it: Google's efficient model
    - gemma2-27b-it: Larger Google model
    
    Requires GROQ_API_KEY environment variable
    Get your API key from: https://console.groq.com/
    """
    
    def __init__(self, model: str = "llama3-8b-8192"):
        self.model = model
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY environment variable is required. "
                "Get your API key from: https://console.groq.com/"
            )
        
        try:
            from groq import Groq
            self.client = Groq(api_key=self.api_key)
        except ImportError:
            raise ImportError(
                "groq package is required for GroqLLMProvider. "
                "Install with: pip install groq"
            )
        
        # Validate model availability
        self._validate_model()
        logger.info(f"Initialized GroqLLMProvider with model {model}")
    
    def _validate_model(self):
        """Validate that the model is available"""
        available_models = [
            "llama3-8b-8192",
            "llama3-70b-8192", 
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
            "gemma2-27b-it"
        ]
        
        if self.model not in available_models:
            logger.warning(
                f"Model {self.model} may not be available. "
                f"Available models: {', '.join(available_models)}"
            )
    
    def generate_response(self, prompt: str, context: Optional[str] = None) -> str:
        """
        Generate response using Groq API with ultra-fast inference
        
        Args:
            prompt: User prompt/question
            context: Optional context to include in the response
            
        Returns:
            Generated response from Groq
            
        Raises:
            Exception: If API call fails
        """
        try:
            messages = []
            
            # Add system context if provided
            if context:
                messages.append({
                    "role": "system",
                    "content": f"Use the following context to answer the question accurately: {context}"
                })
            
            # Add user prompt
            messages.append({
                "role": "user",
                "content": prompt
            })
            
            # Make API call with optimized parameters for speed
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=1000,
                temperature=0.7,  # Balanced creativity and consistency
                top_p=1,          # Use full vocabulary
                stream=False,     # Get complete response
                stop=None         # No stop sequences
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            # Provide helpful error messages
            if "rate limit" in str(e).lower():
                raise Exception("Groq rate limit exceeded. Please try again later.")
            elif "invalid api key" in str(e).lower():
                raise Exception("Invalid Groq API key. Check your GROQ_API_KEY environment variable.")
            elif "model not found" in str(e).lower():
                raise Exception(f"Model {self.model} not available. Try a different model.")
            else:
                raise Exception(f"Groq API error: {e}")
    
    def generate_response_with_context(self, query: str, context_documents: List[str]) -> str:
        """Generate response using context documents"""
        context = "\n\n".join(context_documents)
        return self.generate_response(query, context)
    
    def get_available_models(self) -> List[str]:
        """Get list of available Groq models"""
        return [
            "llama3-8b-8192",
            "llama3-70b-8192", 
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
            "gemma2-27b-it"
        ]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        return {
            "provider": "groq",
            "model": self.model,
            "features": [
                "ultra-fast inference",
                "cost-effective",
                "global edge locations",
                "multiple model options"
            ],
            "speed": "10-100x faster than traditional APIs",
            "use_cases": [
                "real-time applications",
                "high-volume processing",
                "cost-sensitive deployments"
            ]
        }

def get_llm_provider(provider_type: str = "mock", **kwargs) -> LLMProvider:
    """
    Factory function to get appropriate LLM provider
    
    Args:
        provider_type: Type of provider ("mock", "openai", "anthropic", "groq")
        **kwargs: Additional arguments for provider initialization
    
    Returns:
        LLMProvider instance
    """
    if provider_type == "mock":
        return MockLLMProvider(**kwargs)
    elif provider_type == "openai":
        return OpenAILLMProvider(**kwargs)
    elif provider_type == "anthropic":
        return AnthropicLLMProvider(**kwargs)
    elif provider_type == "groq":
        return GroqLLMProvider(**kwargs)
    else:
        raise ValueError(f"Unknown LLM provider type: {provider_type}")

# Example usage
if __name__ == "__main__":
    # Test the LLM providers
    llm = get_llm_provider("mock")
    
    # Test basic response
    response = llm.generate_response("What is artificial intelligence?")
    print(f"Basic response: {response}")
    
    # Test response with context
    context_docs = [
        "AI is a field of computer science focused on creating intelligent machines.",
        "Machine learning is a subset of AI that enables computers to learn without explicit programming."
    ]
    response_with_context = llm.generate_response_with_context(
        "What is the relationship between AI and machine learning?", 
        context_docs
    )
    print(f"Response with context: {response_with_context}")
