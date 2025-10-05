from .core import (
    LocalChatEngine,
    LocalDataIngestion,
    LocalRAGModel,
    LocalEmbedding,
    LocalVectorStore,
    get_system_prompt,
)
from llama_index.core import Settings
from llama_index.core.chat_engine.types import StreamingAgentChatResponse
from llama_index.core.prompts import ChatMessage, MessageRole
from types import SimpleNamespace
import itertools

from types import SimpleNamespace
from llama_index.core import Settings
from rag_chatbot.core.model.model import TokenQueueHandler 

class LocalRAGPipeline:
    def __init__(self, host: str = "host.docker.internal") -> None:
        self._host = host
        self._language = "eng"
        # 1 model mặc định
        self._model_name = "mistralai/mistral-small-3.2-24b-instruct:free"
        self._system_prompt = get_system_prompt("eng", is_rag_prompt=False)
        self._engine = LocalChatEngine(host=host)
        # Luôn set model mặc định ngay từ đầu
        self._default_model = LocalRAGModel.set(host=host)
        self._query_engine = None
        self._ingestion = LocalDataIngestion()
        self._vector_store = LocalVectorStore(host=host)
        Settings.llm = self._default_model
        Settings.embed_model = LocalEmbedding.set(host=host)

    def get_model_name(self):
        # Tên model mặc định
        return self._model_name

    def set_model_name(self, model_name: str):
        # Không cho đổi nữa, chỉ giữ lại để tránh lỗi
        self._model_name = self._model_name

    def get_language(self):
        return self._language

    def set_language(self, language: str):
        self._language = language

    def get_system_prompt(self):
        return self._system_prompt

    def set_system_prompt(self, system_prompt: str | None = None):
        self._system_prompt = system_prompt or get_system_prompt(
            language=self._language, is_rag_prompt=self._ingestion.check_nodes_exist()
        )

    def set_model(self):
        # Luôn set lại bằng model mặc định
        Settings.llm = LocalRAGModel.set(
            host=self._host,
        )
        self._default_model = Settings.llm

    def reset_engine(self):
        self._query_engine = self._engine.set_engine(
            llm=self._default_model, nodes=[], language=self._language
        )

    def reset_documents(self):
        self._ingestion.reset()

    def clear_conversation(self):
        self._query_engine.reset()

    def reset_conversation(self):
        self.reset_engine()
        self.set_system_prompt(
            get_system_prompt(language=self._language, is_rag_prompt=False)
        )

    def set_embed_model(self, model_name: str):
        Settings.embed_model = LocalEmbedding.set(model_name, self._host)

    def pull_model(self, model_name: str):
        # Không còn pull từ Ollama, chỉ return None
        return None

    # def pull_embed_model(self, model_name: str):
    #     return LocalEmbedding.pull(self._host, model_name)

    def check_exist(self, model_name: str) -> bool:
        # Luôn coi như model đã tồn tại
        return True

    # def check_exist_embed(self, model_name: str) -> bool:
    #     return LocalEmbedding.check_model_exist(self._host, model_name)

    def store_nodes(self, input_files: list[str] = None) -> None:
        self._ingestion.store_nodes(input_files=input_files)

    def set_chat_mode(self, system_prompt: str | None = None):
        self.set_language(self._language)
        self.set_system_prompt(system_prompt)
        self.set_model()
        self.set_engine()

    def set_engine(self):
        self._query_engine = self._engine.set_engine(
            llm=self._default_model,
            nodes=self._ingestion.get_ingested_nodes(),
            language=self._language,
        )

    def get_history(self, chatbot: list[list[str]]):
        history = []
        for chat in chatbot:
            if chat[0]:
                history.append(ChatMessage(role=MessageRole.USER, content=chat[0]))
                history.append(ChatMessage(role=MessageRole.ASSISTANT, content=chat[1]))
        return history

    def query(self, mode: str, message: str, chatbot: list[list[str]]):
        if self._query_engine is None:
            self.set_engine()

        # 1) Tạo bridge handler và gắn vào Settings.callback_manager
        cbm = Settings.callback_manager
        bridge = TokenQueueHandler(cbm, name="BRIDGE")
        cbm.add_handler(bridge)

        # 2) Gọi stream_chat của ChatEngine để nó chạy và phát LLM_TOKEN vào callback
        if mode == "chat":
            history = self.get_history(chatbot)
            _ = self._query_engine.stream_chat(message, history)
        else:
            self._query_engine.reset()
            _ = self._query_engine.stream_chat(message)

        # 3) Trả về object có .response_gen là generator của bridge
        #    (ChatEngine.response_gen có thể rỗng ở version hiện tại)
        return SimpleNamespace(response_gen=bridge.iter_tokens())