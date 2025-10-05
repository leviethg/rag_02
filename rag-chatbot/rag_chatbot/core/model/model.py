from llama_index.core.llms import CustomLLM
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseGen,
    LLMMetadata,
    CompletionResponse,
    CompletionResponseGen
)
from llama_index.core.callbacks import CBEventType, EventPayload
from pydantic import Field
from ...setting import RAGSettings
import requests
import asyncio
import json
from llama_index.core.callbacks import CBEventType, EventPayload

# Tap toàn bộ callback
# model.py

try:
    from llama_index.core.callbacks import CBEventType, EventPayload
except Exception:
    CBEventType = None
    EventPayload = None

class TapAllEventsHandler:
    """Logger đơn giản để bắt mọi callback từ CallbackManager (duck-typed)."""
    def __init__(self, name="CB"):
        self.name = name
        # 🔧 BẮT BUỘC CHO CallbackManager (nếu không sẽ AttributeError)
        # là các tập event muốn bỏ qua; mặc định để trống
        self.event_starts_to_ignore = set()
        self.event_ends_to_ignore = set()
        # một số bản còn tham chiếu tới cặp start/end cần bỏ qua:
        self.event_pairs_to_ignore = set()
        # có thể có cờ enabled
        self.enabled = True

    # ==== trace ====
    def start_trace(self, trace_id=None, **kwargs):
        print(f"[{self.name}] TRACE START id={trace_id}")

    def end_trace(self, trace_id=None, trace_map=None, **kwargs):
        size = len(trace_map) if isinstance(trace_map, dict) else None
        print(f"[{self.name}] TRACE END   id={trace_id} trace_map_size={size}")

    # ==== Event API (mới) ====
    def on_event_start(self, event_type, payload=None, event_id=None, parent_id=None, **kwargs):
        keys = list((payload or {}).keys())
        et = getattr(event_type, "name", str(event_type))
        print(f"[{self.name}] start {et} keys={keys} event_id={event_id} parent_id={parent_id}")

    def on_event_end(self, event_type, payload=None, event_id=None, parent_id=None, **kwargs):
        keys = list((payload or {}).keys())
        et = getattr(event_type, "name", str(event_type))
        print(f"[{self.name}] end   {et} keys={keys} event_id={event_id} parent_id={parent_id}")

    # ==== Legacy API (cũ) – bản bạn dùng có thể KHÔNG gọi ====
    def on_llm_new_token(self, token=None, **kwargs):
        print(f"[{self.name}] token: {token!r}")



# model.py
from queue import Queue

try:
    from llama_index.core.callbacks import CBEventType, EventPayload
except Exception:
    CBEventType = None
    EventPayload = None

def _is_event(ev, name: str) -> bool:
    # So khớp cả enum lẫn string
    if ev is None:
        return False
    if hasattr(ev, "name"):
        return ev.name == name
    return str(ev) == name

class TokenQueueHandler:
    """Bridge LLM_TOKEN -> queue để UI iterate."""
    def __init__(self, manager, name="BRIDGE"):
        self.manager = manager
        self.name = name
        self.q = Queue()
        self._SENTINEL = object()
        # các thuộc tính mà CallbackManager kỳ vọng
        self.event_starts_to_ignore = set()
        self.event_ends_to_ignore = set()
        self.event_pairs_to_ignore = set()
        self.enabled = True

    # trace (no-op)
    def start_trace(self, trace_id=None, **kwargs): pass
    def end_trace(self, trace_id=None, trace_map=None, **kwargs): pass

    # Event API
    def on_event_start(self, event_type, payload=None, event_id=None, parent_id=None, **kwargs):
        # gom token
        if _is_event(event_type, "LLM_TOKEN"):
            token_key = getattr(EventPayload, "TOKEN", None) or "token"
            tok = (payload or {}).get(token_key)
            if isinstance(tok, str) and tok:
                self.q.put(tok)

    def on_event_end(self, event_type, payload=None, event_id=None, parent_id=None, **kwargs):
        # kết thúc phiên LLM -> đóng generator
        if _is_event(event_type, "LLM"):
            self.q.put(self._SENTINEL)
            try:
                # tự tháo handler để không bị lặp ở phiên sau
                self.manager.remove_handler(self)
            except Exception:
                pass

    # generator cho UI
    def iter_tokens(self):
        while True:
            item = self.q.get()
            if item is self._SENTINEL:
                break
            yield item  # yield string để UI concat


class OpenRouterLLM(CustomLLM):
    """Custom LLM wrapper for OpenRouter"""

    api_key: str = Field(..., description="API key for OpenRouter")
    model: str = Field(..., description="Model name on OpenRouter")
    # base_url: str = Field(default="https://openrouter.ai/api/v1/chat/completions")
    base_url: str = "https://openrouter.ai/api/v1/chat/completions"

    # -------- metadata --------
    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=16384,
            num_output=2048,
            model_name=self.model
        )

    # -------- completion --------
    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {"model": self.model, "messages": [{"role": "user", "content": prompt}]}
        r = requests.post(self.base_url, json=payload, headers=headers)
        r.raise_for_status()
        data = r.json()
        text = data["choices"][0]["message"]["content"]
        return CompletionResponse(text=text)

    async def acomplete(self, prompt: str, **kwargs) -> CompletionResponse:
        return await asyncio.to_thread(self.complete, prompt, **kwargs)

    # -------- stream completion --------
    def stream_complete(self, prompt: str, **kwargs) -> CompletionResponseGen:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {"model": self.model, "messages": [{"role": "user", "content": prompt}], "stream": True}
        with requests.post(self.base_url, json=payload, headers=headers, stream=True) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line or line.startswith(b"data: [DONE]"):
                    continue
                data = line.decode("utf-8").removeprefix("data: ")
                delta = json.loads(data)
                if "choices" in delta and delta["choices"][0]["delta"].get("content"):
                    yield CompletionResponse(text=delta["choices"][0]["delta"]["content"])

    async def astream_complete(self, prompt: str, **kwargs) -> CompletionResponseGen:
        async def gen():
            for chunk in self.stream_complete(prompt, **kwargs):
                yield chunk
        return gen()

    # -------- chat --------
    def chat(self, messages, **kwargs) -> ChatResponse:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        # payload = {"model": self.model, "messages": messages}
        payload = {"model": self.model, "messages": self._format_messages(messages)}
        r = requests.post(self.base_url, json=payload, headers=headers)
        r.raise_for_status()
        data = r.json()
        content = data["choices"][0]["message"]["content"]
        return ChatResponse(message=ChatMessage(role="assistant", content=content))

    async def achat(self, messages, **kwargs) -> ChatResponse:
        return await asyncio.to_thread(self.chat, messages, **kwargs)

    # model.py - trong class OpenRouterLLM

    # model.py
    

    def stream_chat(self, messages, **kwargs) -> ChatResponseGen:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": self._format_messages(messages),
            "stream": True,
        }

        # ✅ import enum an toàn theo version (có thể None)
        try:
            from llama_index.core.callbacks import CBEventType, EventPayload
        except Exception:
            CBEventType = None
            EventPayload = None

        # ---------- THAY helper này ----------
        def _emit_token(tok: str):
            from llama_index.core import Settings
            cb_llm = getattr(self, "callback_manager", None)
            cb_global = getattr(Settings, "callback_manager", None)

            managers = []
            if cb_llm:
                managers.append(("LLM", cb_llm))
            if cb_global and cb_global is not cb_llm:
                managers.append(("Settings", cb_global))

            #print(f"[DBG] _emit_token tok={tok!r} managers={[(name, id(cb)) for name, cb in managers]}")
            if not managers:
                print("[DBG] no managers to emit to")
                return

            for name, cb in managers:
                # Event API (đừng return sớm)
                try:
                    evt = getattr(CBEventType, "LLM_TOKEN", None) or "LLM_TOKEN"
                    key = getattr(EventPayload, "TOKEN", None) or "token"
                    cb.on_event_start(evt, payload={key: tok})
                    cb.on_event_end(evt, payload={key: tok})
                    #print(f"[DBG]    OK event api on {name}")
                except Exception as e:
                    print(f"[DBG]    FAIL event api on {name}: {e!r}")

                # Legacy API
                try:
                    cb.on_llm_new_token(token=tok)
                    #print(f"[DBG]    OK legacy api on {name}")
                except Exception as e:
                    print(f"[DBG]    FAIL legacy api on {name}: {e!r}")

        # ---------- THAY helper này ----------
        def _emit_llm_event(start: bool):
            from llama_index.core import Settings
            cb_llm = getattr(self, "callback_manager", None)
            cb_global = getattr(Settings, "callback_manager", None)

            managers = []
            if cb_llm:
                managers.append(("LLM", cb_llm))
            if cb_global and cb_global is not cb_llm:
                managers.append(("Settings", cb_global))

            #print(f"[DBG] _emit_llm_event start={start} managers={[(name, id(cb)) for name, cb in managers]}")

            for name, cb in managers:
                # Event API
                try:
                    evt = getattr(CBEventType, "LLM", None) or "LLM"
                    event_payload = {}
                    if start:
                        key_model = getattr(EventPayload, "MODEL", None)
                        key_msgs = getattr(EventPayload, "MESSAGES", None)
                        if key_model:
                            event_payload[key_model] = self.model
                        if key_msgs:
                            event_payload[key_msgs] = payload["messages"]
                        cb.on_event_start(evt, payload=event_payload)
                    else:
                        cb.on_event_end(evt, payload=None)
                    #print(f"[DBG]    OK event LLM on {name}")
                except Exception as e:
                    print(f"[DBG]    FAIL event LLM on {name}: {e!r}")

                # Legacy API
                try:
                    if start:
                        cb.on_llm_start(prompt=payload)  # prompt cho 1 số bản
                    else:
                        cb.on_llm_end()
                    #print(f"[DBG]    OK legacy LLM on {name}")
                except Exception as e:
                    print(f"[DBG]    FAIL legacy LLM on {name}: {e!r}")

        # --------- phần còn lại giữ nguyên ----------
        with requests.post(self.base_url, json=payload, headers=headers, stream=True) as r:
            r.raise_for_status()
            _emit_llm_event(start=True)

            for line in r.iter_lines():
                if not line:
                    continue

                raw = line.decode("utf-8").strip()
                #print(f"📥 Nhận chunk: {raw}")

                if raw.startswith(":") or raw == "" or raw.startswith("data: [DONE]"):
                    continue
                if not raw.startswith("data: "):
                    continue

                raw = raw.removeprefix("data: ").strip()
                try:
                    delta = json.loads(raw)
                except json.JSONDecodeError as e:
                    print(f"❌ JSONDecodeError: {e} | chunk={raw!r}")
                    continue

                if "choices" in delta:
                    content = delta["choices"][0].get("delta", {}).get("content")
                    if content:
                        _emit_token(content)
                        yield ChatResponse(
                            message=ChatMessage(role="assistant", content=content)
                        )

            _emit_llm_event(start=False)
    
    # def stream_chat(self, messages, **kwargs) -> ChatResponseGen:
    #     import time, random
    #     from requests.exceptions import (
    #         ChunkedEncodingError, ConnectionError, ReadTimeout, HTTPError
    #     )
    #     from urllib3.exceptions import ProtocolError
    #     from http.client import IncompleteRead

    #     headers = {
    #         "Authorization": f"Bearer {self.api_key}",
    #         "Content-Type": "application/json",
    #     }
    #     payload = {
    #         "model": self.model,
    #         "messages": self._format_messages(messages),
    #         "stream": True,
    #     }

    #     # ✅ import enum an toàn theo version (có thể None) — GIỮ NGUYÊN
    #     try:
    #         from llama_index.core.callbacks import CBEventType, EventPayload
    #     except Exception:
    #         CBEventType = None
    #         EventPayload = None

    #     # ---------- GIỮ NGUYÊN helper của bạn ----------
    #     def _emit_token(tok: str):
    #         from llama_index.core import Settings
    #         cb_llm = getattr(self, "callback_manager", None)
    #         cb_global = getattr(Settings, "callback_manager", None)

    #         managers = []
    #         if cb_llm:
    #             managers.append(("LLM", cb_llm))
    #         if cb_global and cb_global is not cb_llm:
    #             managers.append(("Settings", cb_global))

    #         if not managers:
    #             return

    #         for name, cb in managers:
    #             # Event API
    #             try:
    #                 evt = getattr(CBEventType, "LLM_TOKEN", None) or "LLM_TOKEN"
    #                 key = getattr(EventPayload, "TOKEN", None) or "token"
    #                 cb.on_event_start(evt, payload={key: tok})
    #                 cb.on_event_end(evt, payload={key: tok})
    #             except Exception:
    #                 pass
    #             # Legacy API
    #             try:
    #                 cb.on_llm_new_token(token=tok)
    #             except Exception:
    #                 pass

    #     # ---------- GIỮ NGUYÊN helper của bạn ----------
    #     def _emit_llm_event(start: bool):
    #         from llama_index.core import Settings
    #         cb_llm = getattr(self, "callback_manager", None)
    #         cb_global = getattr(Settings, "callback_manager", None)

    #         managers = []
    #         if cb_llm:
    #             managers.append(("LLM", cb_llm))
    #         if cb_global and cb_global is not cb_llm:
    #             managers.append(("Settings", cb_global))

    #         for name, cb in managers:
    #             # Event API
    #             try:
    #                 evt = getattr(CBEventType, "LLM", None) or "LLM"
    #                 event_payload = {}
    #                 if start:
    #                     key_model = getattr(EventPayload, "MODEL", None)
    #                     key_msgs = getattr(EventPayload, "MESSAGES", None)
    #                     if key_model:
    #                         event_payload[key_model] = self.model
    #                     if key_msgs:
    #                         event_payload[key_msgs] = payload["messages"]
    #                     cb.on_event_start(evt, payload=event_payload)
    #                 else:
    #                     cb.on_event_end(evt, payload=None)
    #             except Exception:
    #                 pass
    #             # Legacy API
    #             try:
    #                 if start:
    #                     cb.on_llm_start(prompt=payload)
    #                 else:
    #                     cb.on_llm_end()
    #             except Exception:
    #                 pass

    #     # ------------- helper nhỏ: log rate limit -------------
    #     def _log_rate_headers(resp):
    #         try:
    #             print("[RATE] status=", resp.status_code)
    #             for k, v in resp.headers.items():
    #                 lk = k.lower()
    #                 if lk.startswith("x-ratelimit") or lk in ("retry-after",):
    #                     print(f"[RATE] {k}: {v}")
    #         except Exception:
    #             pass

    #     # ------------- helper: parse Retry-After/reset ----------
    #     def _parse_retry_after(headers: dict) -> float | None:
    #         ra = headers.get("Retry-After") or headers.get("retry-after")
    #         if ra:
    #             try:
    #                 return float(ra)  # giây
    #             except Exception:
    #                 pass
    #         rst = headers.get("X-RateLimit-Reset") or headers.get("x-ratelimit-reset")
    #         if rst:
    #             try:
    #                 v = float(rst)
    #                 now = time.time()
    #                 # nếu server trả epoch tương lai → chờ tới thời điểm đó
    #                 return max(0.0, v - now) if v > now else max(0.0, v)
    #             except Exception:
    #                 pass
    #         return None

    #     # ------------- helper: backoff + jitter -----------------
    #     def _sleep_backoff(attempt: int, resp=None):
    #         wait = _parse_retry_after(getattr(resp, "headers", {})) if resp is not None else None
    #         if wait is None:
    #             wait = min(BACKOFF_CAP, (BACKOFF_BASE ** attempt) + random.uniform(0, 0.5))
    #         try:
    #             _emit_token(f"\n[retry in {wait:.1f}s]\n")
    #         except Exception:
    #             pass
    #         time.sleep(wait)

    #     # ================= Retry config =================
    #     MAX_ATTEMPTS    = 6         # số lần thử tối đa
    #     DEADLINE_SEC    = 90        # tổng thời gian cho chuỗi retry
    #     BACKOFF_BASE    = 1.8        # cơ số backoff mũ
    #     BACKOFF_CAP     = 10.0       # trần backoff
    #     NONSTREAM_AFTER = 3          # sau X lần fail → fallback non-stream

    #     start_wall      = time.time()
    #     attempt         = 0
    #     llm_started     = False
    #     stream_produced = False

    #     # ================= Vòng retry ===================
    #     while attempt < MAX_ATTEMPTS and (time.time() - start_wall) < DEADLINE_SEC:
    #         r = None
    #         try:
    #             r = requests.post(
    #                 self.base_url,
    #                 json=payload,
    #                 headers=headers,
    #                 stream=True,
    #                 timeout=(10, 600),  # (connect, read)
    #             )

    #             _log_rate_headers(r)

    #             # Các lỗi không nên retry nhiều lần
    #             if r.status_code in (400, 401, 403, 404):
    #                 r.raise_for_status()

    #             # 429 → tôn trọng Retry-After / RateLimit headers
    #             if r.status_code == 429:
    #                 _sleep_backoff(attempt, r)
    #                 attempt += 1
    #                 try:
    #                     r.close()
    #                 except Exception:
    #                     pass

    #                 # Fallback non-stream sau vài lần thất bại
    #                 if attempt >= NONSTREAM_AFTER:
    #                     try:
    #                         _emit_token("\n[switching to non-stream completion]\n")
    #                     except Exception:
    #                         pass
    #                     r2 = requests.post(
    #                         self.base_url,
    #                         json={**payload, "stream": False},
    #                         headers=headers,
    #                         timeout=(10, 120),
    #                     )
    #                     _log_rate_headers(r2)
    #                     r2.raise_for_status()
    #                     data = r2.json()
    #                     text = (
    #                         data.get("choices", [{}])[0]
    #                             .get("message", {})
    #                             .get("content", "")
    #                     )
    #                     if text:
    #                         _emit_token(text)
    #                         yield ChatResponse(
    #                             message=ChatMessage(role="assistant", content=text)
    #                         )
    #                     # kết thúc phiên
    #                     if llm_started:
    #                         _emit_llm_event(start=False)
    #                     return
    #                 # chưa fallback → thử lại
    #                 continue

    #             # Thành công → mở stream
    #             r.raise_for_status()
    #             if not llm_started:
    #                 _emit_llm_event(start=True)
    #                 llm_started = True

    #             with r:
    #                 for line in r.iter_lines(decode_unicode=False):
    #                     if not line:
    #                         continue

    #                     raw = line.decode("utf-8", errors="replace").strip()
    #                     # if cần: print(f"📥 Nhận chunk: {raw}")

    #                     if raw.startswith(":") or raw == "":
    #                         continue
    #                     if raw.startswith("data: [DONE]"):
    #                         break
    #                     if not raw.startswith("data: "):
    #                         continue

    #                     body = raw[6:].strip()
    #                     try:
    #                         delta = json.loads(body)
    #                     except json.JSONDecodeError:
    #                         continue

    #                     content = (
    #                         delta.get("choices", [{}])[0]
    #                             .get("delta", {})
    #                             .get("content")
    #                     )
    #                     if content:
    #                         stream_produced = True
    #                         _emit_token(content)
    #                         yield ChatResponse(
    #                             message=ChatMessage(role="assistant", content=content)
    #                         )
    #             # stream kết thúc bình thường → thoát vòng retry
    #             break

    #         except (ChunkedEncodingError, ProtocolError, IncompleteRead, ConnectionError, ReadTimeout) as e:
    #             # lỗi mạng / EOF / timeout → backoff & retry
    #             try:
    #                 _emit_token(f"\n[network issue: {type(e).__name__} — retrying]\n")
    #             except Exception:
    #                 pass
    #             _sleep_backoff(attempt, r)
    #             attempt += 1
    #             continue

    #         except HTTPError as e:
    #             # HTTP lỗi khác (đã raise ở trên) → ném tiếp
    #             code = getattr(e.response, "status_code", None)
    #             try:
    #                 _emit_token(f"\n[http error {code} — abort]\n")
    #             except Exception:
    #                 pass
    #             raise

    #         finally:
    #             try:
    #                 if r is not None and not r.raw.closed:
    #                     r.close()
    #             except Exception:
    #                 pass

    #     # ================= finalize phiên =================
    #     try:
    #         if llm_started:
    #             _emit_llm_event(start=False)
    #     except Exception:
    #         pass

    #     if not stream_produced:
    #         _emit_token("\n[stopped after retry budget exceeded]\n")

            
                    
    async def astream_chat(self, messages, **kwargs) -> ChatResponseGen:
        async def gen():
            for chunk in self.stream_chat(messages, **kwargs):
                yield chunk
        return gen()
    
    def _format_messages(self, messages):
        formatted = []
        for m in messages:
            if isinstance(m, ChatMessage):
                # ép Enum về string
                role = m.role.value if hasattr(m.role, "value") else str(m.role)
                formatted.append({"role": role, "content": m.content})
            elif isinstance(m, dict):
                role = m.get("role")
                if hasattr(role, "value"):  # nếu vẫn còn Enum
                    role = role.value
                formatted.append({"role": role, "content": m.get("content")})
            else:
                raise TypeError(f"Unsupported message type: {type(m)}")
        return formatted




class LocalRAGModel:
    def __init__(self) -> None:
        pass

    @staticmethod
    def set(
        model_name: str | None = None,
        system_prompt: str | None = None,
        host: str = "localhost",
        setting: RAGSettings | None = None,
    ):
        """Always return OpenRouterLLM — bỏ hoàn toàn Ollama/OpenAI gốc."""
        setting = setting or RAGSettings()
        return OpenRouterLLM(
            api_key=setting.openrouter.api_key,
            model=model_name or setting.openrouter.llm
        )

    @staticmethod
    def pull(host: str, model_name: str):
        return None

    @staticmethod
    def check_model_exist(host: str, model_name: str) -> bool:
        return True
