from llama_index.core.chat_engine import CondensePlusContextChatEngine, SimpleChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.llms.llm import LLM
from llama_index.core.schema import BaseNode
from typing import List
from .retriever import LocalRetriever
from ...setting import RAGSettings
from llama_index.core import Settings
from rag_chatbot.core.model.model import TapAllEventsHandler



class LocalChatEngine:
    def __init__(
        self,
        setting: RAGSettings | None = None,
        host: str = "host.docker.internal"
    ):
        super().__init__()
        self._setting = setting or RAGSettings()
        self._retriever = LocalRetriever(self._setting)
        self._host = host

    def set_engine(
        self,
        llm: LLM,
        nodes: List[BaseNode],
        language: str = "eng",
    ) -> CondensePlusContextChatEngine | SimpleChatEngine:

        # >>> GẮN TAP HANDLER TRƯỚC KHI KHỞI TẠO CHAT ENGINE <<<
        tap = TapAllEventsHandler("CB")
        Settings.callback_manager.add_handler(tap)

        if getattr(llm, "callback_manager", None) and llm.callback_manager is not Settings.callback_manager:
            llm.callback_manager.add_handler(tap)

        # In id để đối chiếu LLM vs Settings callback_manager
        # print("[CB] ids  Settings:", id(Settings.callback_manager))
        # if getattr(llm, "callback_manager", None):
        #     print("[CB] ids  LLM     :", id(llm.callback_manager))

        # Chọn callback_manager dùng cho ChatEngine: ưu tiên cái global cho đồng bộ
        cbm = Settings.callback_manager
        
        # Normal chat engine
        if len(nodes) == 0:
            return SimpleChatEngine.from_defaults(
                llm=llm,
                memory=ChatMemoryBuffer(
                    token_limit=self._setting.openrouter.chat_token_limit

                ),
                callback_manager=cbm,
            )

        # Chat engine with documents
        retriever = self._retriever.get_retrievers(
            llm=llm,
            language=language,
            nodes=nodes
        )
        return CondensePlusContextChatEngine.from_defaults(
            retriever=retriever,
            llm=llm,
            memory=ChatMemoryBuffer(
                token_limit=self._setting.openrouter.chat_token_limit
            ),
            callback_manager=cbm,
        )
