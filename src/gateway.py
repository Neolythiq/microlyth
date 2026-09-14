from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any

from pathlib import Path

from microlyth.src.helpers import GetKeyFromRegistry

# --- 1. CAPABILITY INTERFACES ---

class TextGenerationModel(ABC):
    @abstractmethod
    def SendRequest(
        self, 
        prompt: str, 
        systemPrompt: Optional[str] = None,
        **kwargs: Any
    ) -> str:
        pass

class EmbeddingModel(ABC):
    @abstractmethod
    def GetEmbedding(self, text: str) -> List[float]:
        pass

class RerankModel(ABC):
    @abstractmethod
    def Rerank(self, query: str, documents: List[str], topN: int) -> List[Dict[str, Any]]:
        pass

class VisionModel(ABC):
    @abstractmethod
    def AnalyzeImage(self, imagePath: str, prompt: str) -> str:
        pass

class GenAIGateway:
    """
    Central router that manages model instances by capability and tier/name.
    """
    def __init__(self):
        self._llms: Dict[str, TextGenerationModel] = {}
        self._embeddings: Dict[str, EmbeddingModel] = {}
        self._rerankers: Dict[str, RerankModel] = {}
        self._vision_models: Dict[str, VisionModel] = {}

    def RegisterLLM(self, name: str, client: TextGenerationModel) -> None:
        self._llms[name] = client

    def RegisterEmbedding(self, name: str, client: EmbeddingModel) -> None:
        self._embeddings[name] = client

    def RegisterReranker(self, name: str, client: RerankModel) -> None:
        self._rerankers[name] = client

    def RegisterVisionModel(self, name: str, client: VisionModel) -> None:
        self._vision_models[name] = client

    # Routing execution methods
    def GenerateText(self, prompt: str, modelName: str = "default", **kwargs) -> str:
        if modelName not in self._llms:
            raise KeyError(f"LLM '{modelName}' is not registered.")
        return self._llms[modelName].SendRequest(prompt, **kwargs)

    def GetEmbedding(self, text: str, modelName: str = "default") -> List[float]:
        if modelName not in self._embeddings:
            raise KeyError(f"Embedding model '{modelName}' is not registered.")
        return self._embeddings[modelName].GetEmbedding(text)

    def Rerank(self, query: str, documents: List[str], modelName: str = "default", topN: int = 3) -> List[Dict[str, Any]]:
        if modelName not in self._rerankers:
            raise KeyError(f"Reranker model '{modelName}' is not registered.")
        return self._rerankers[modelName].Rerank(query, documents, topN)

    def AnalyzeImage(self, imagePath: str, prompt: str, modelName: str = "default") -> str:
        if modelName not in self._vision_models:
            raise KeyError(f"Vision model '{modelName}' is not registered.")
        return self._vision_models[modelName].AnalyzeImage(imagePath, prompt)

class UniversalClient(TextGenerationModel, EmbeddingModel):
    def __init__(self,
                 endPoint: str = None,
                 textGenModelEndpoint: str = None,
                 textGenModelApiKey: str = None,
                 textGenModelName: str = None,
                 embeddingModelEndpoint: str = None,
                 embeddingModelApiKey: str = None,
                 embeddingModelName: str = None,
                 maxTokens=3000,
                 temperature=0.7):
        from openai import OpenAI

        if endPoint is not None and textGenModelEndpoint is None:
            textGenModelEndpoint = endPoint
        if endPoint is not None and embeddingModelEndpoint is None:
            embeddingModelEndpoint = endPoint

        self.textGenModelEndpoint = textGenModelEndpoint or GetKeyFromRegistry("textGenModelEndpoint")
        self.textGenModelApiKey = textGenModelApiKey or GetKeyFromRegistry("openAIapiKey")
        self.textGenModelName = textGenModelName or GetKeyFromRegistry("mainModel")
        self.embeddingModelEndpoint = embeddingModelEndpoint or GetKeyFromRegistry("embeddingModelEndpoint")
        self.embeddingModelApiKey = embeddingModelApiKey or GetKeyFromRegistry("embeddingModelApiKey")
        self.embeddingModelName = embeddingModelName or GetKeyFromRegistry("embeddingModel")

        self.maxTokens = maxTokens
        self.temperature = temperature
        
        self.textGenClient = OpenAI(
            base_url=self.textGenModelEndpoint,
            api_key=self.textGenModelApiKey,
        )
        self.embeddingClient = OpenAI(
            base_url=self.embeddingModelEndpoint,
            api_key=self.embeddingModelApiKey,
        )

    def SendRequest(self, prompt : str, 
                    systemPrompt: Optional[str] =None,
                    taskType: Optional[str] = None,
                    targetTier: Optional[str] = None):
        response = self.textGenClient.chat.completions.create(
            model=self.textGenModelName,
            messages=[
                {"role": "system", "content": systemPrompt} if systemPrompt else {},
                {"role": "user", "content": prompt}
            ],
            max_tokens=self.maxTokens,
            temperature=self.temperature
        )
        return response.choices[0].message.content.strip()

    def GetEmbedding(self, text):
        response = self.embeddingClient.embeddings.create(
            input=text,
            model=self.embeddingModelName
        )
        return response.data[0].embedding

class OllamaTextGenClient(TextGenerationModel):
    def __init__(self,
                 endPoint: str = "http://localhost:11434",
                 modelName:str = None,
                 maxTokens: int = 3000, 
                 temperature: float = 0.7):
        import ollama

        # Gather your environment keys directly from the base gateway manager registry
        self.endPoint = endPoint or GetKeyFromRegistry("ollamaEndPoint")
        if not modelName:
            self.model = GetKeyFromRegistry("localDefaultTextGenModel")       # e.g., "gemma4" or "llama3.1"
        else:
            self.model = modelName

        self.maxTokens = maxTokens
        self.temperature = temperature
        
        self.client = ollama.Client(host=self.endPoint)

    def SendRequest(self, 
                    prompt : str, 
                    systemPrompt: Optional[str] =None,
                    taskType: Optional[str] = None,
                    targetTier: Optional[str] = None) -> str:
        messageChain = []
        
        if systemPrompt:
            messageChain.append({"role": "system", "content": systemPrompt})
            
        messageChain.append({"role": "user", "content": prompt})
        
        response = self.client.chat(
            model=self.model,
            messages=messageChain,
            options={
                "temperature": self.temperature,
                "num_predict": self.maxTokens
            }
        )
        
        return response["message"]["content"].strip()

class OllamaVisionClient(VisionModel):
    def __init__(self,
                 endPoint: str = "http://localhost:11434",
                 modelName:str = None,
                 maxTokens: int = 3000, 
                 temperature: float = 0.7):
        import ollama

        # Gather your environment keys directly from the base gateway manager registry
        self.endPoint = endPoint or GetKeyFromRegistry("ollamaEndPoint")
        if not modelName:
            self.model = GetKeyFromRegistry("localDefaultVisionModel")       # e.g., "gemma4" or "llama3.1"
        else:
            self.model = modelName

        self.maxTokens = maxTokens
        self.temperature = temperature
        
        self.client = ollama.Client(host=self.endPoint)

    def AnalyzeImage(self, imagePath: str, prompt: str, systemPrompt: Optional[str] = None) -> str: 
        messageChain = []
        if systemPrompt:
            messageChain.append({"role": "system", "content": systemPrompt})
        messageChain.append({"role": "user", "content": prompt})
        messageChain.append({"role": "user", "images": [imagePath]})

        response = self.client.chat(
            model=self.model,
            messages=messageChain,
            options={
                "temperature": self.temperature,
                "num_predict": self.maxTokens
            }
        )

        return response["message"]["content"].strip()

class LocalBGEM3MClient(EmbeddingModel):
    def __init__(self,
                 endPoint: str = None,
                 modelName:str = "bgem3M",
                 maxTokens: int = 3000):

        if endPoint is None:
            self.endPoint = GetKeyFromRegistry("bgem3MEndPoint")
        else:
            self.endPoint = endPoint

        self.maxTokens = maxTokens

        isLocal = Path(endPoint).exists()
        from FlagEmbedding import BGEM3FlagModel
        if isLocal:
            self.model = BGEM3FlagModel(self.endPoint, use_fp16=True)
        else:
            self.model = None  # or some remote initialization if needed

    def GetEmbedding(self, 
                     text: str | list[str],
                     dense = True) -> List[float]:
        if self.model is None:
            raise ValueError("Local model is not initialized.")
        return self.model.encode(
            text if isinstance(text, list) else [text],
            return_dense=dense,         # Dense vectors
            return_sparse=not dense     # Sparse lexical weights
        )