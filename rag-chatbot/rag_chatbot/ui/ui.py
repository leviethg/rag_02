import os
import shutil
import json
import sys
import time
import gradio as gr
from dataclasses import dataclass
from typing import ClassVar
from llama_index.core.chat_engine.types import StreamingAgentChatResponse
from .theme import JS_LIGHT_THEME, CSS
from ..pipeline import LocalRAGPipeline
from ..logger import Logger


@dataclass
class DefaultElement:
    DEFAULT_MESSAGE: ClassVar[dict] = {"text": ""}
    #DEFAULT_MESSAGE: ClassVar[str] = ""

    DEFAULT_MODEL: str = ""
    DEFAULT_HISTORY: ClassVar[list] = []
    DEFAULT_DOCUMENT: ClassVar[list] = []

    HELLO_MESSAGE: str = "Hi 👋, how can I help you today?"
    SET_MODEL_MESSAGE: str = "You need to choose LLM model 🤖 first!"
    EMPTY_MESSAGE: str = "You need to enter your message!"
    DEFAULT_STATUS: str = "Ready!"
    # CONFIRM_PULL_MODEL_STATUS: str = "Confirm Pull Model!"
    # PULL_MODEL_SCUCCESS_STATUS: str = "Pulling model 🤖 completed!"
    # PULL_MODEL_FAIL_STATUS: str = "Pulling model 🤖 failed!"
    # MODEL_NOT_EXIST_STATUS: str = "Model doesn't exist!"
    PROCESS_DOCUMENT_SUCCESS_STATUS: str = "Processing documents 📄 completed!"
    PROCESS_DOCUMENT_EMPTY_STATUS: str = "Empty documents!"
    ANSWERING_STATUS: str = "Answering!"
    COMPLETED_STATUS: str = "Completed!"


class LLMResponse:
    def __init__(self) -> None:
        pass

    def _yield_string(self, message: str):
        for i in range(len(message)):
            time.sleep(0.01)
            yield (
                DefaultElement.DEFAULT_MESSAGE,
                # [[None, message[: i + 1]]],
                [["", message[: i + 1]]],
                DefaultElement.DEFAULT_STATUS,
            )

    def welcome(self):
        yield from self._yield_string(DefaultElement.HELLO_MESSAGE)

    def set_model(self):
        yield from self._yield_string(DefaultElement.SET_MODEL_MESSAGE)

    def empty_message(self):
        yield from self._yield_string(DefaultElement.EMPTY_MESSAGE)

    def stream_response(
        self,
        message: dict[str, str] | str,
        history: list[list[str]],
        response: StreamingAgentChatResponse,
    ):
        #print("[UI] >>> ENTER stream_response")
        #print(f"[UI] response.response_gen type={type(response.response_gen)} repr={response.response_gen!r}")

        user_msg = message["text"] if isinstance(message, dict) else str(message)

        def _tap(gen):
            i = 0
            for item in gen:
                has_attrs = [a for a in ("message", "delta", "content", "event", "data", "text") if hasattr(item, a)]
                #print(f"[UI] TAP upstream chunk {i}: type={type(item)} has={has_attrs} repr={item!r}")
                i += 1
                yield item

        def _extract_text(piece) -> str:
            if isinstance(piece, str):
                return piece
            if isinstance(piece, dict):
                d = piece
                if "delta" in d:
                    delta = d["delta"]
                    if isinstance(delta, dict) and isinstance(delta.get("content"), str):
                        return delta["content"]
                    if isinstance(delta, str):
                        return delta
                if isinstance(d.get("content"), str):
                    return d["content"]
                if isinstance(d.get("text"), str):
                    return d["text"]
                return ""
            msg = getattr(piece, "message", None)
            if msg is not None:
                c = getattr(msg, "content", None)
                if isinstance(c, str):
                    return c
            delta = getattr(piece, "delta", None)
            if isinstance(delta, str):
                return delta
            if isinstance(delta, dict):
                c = delta.get("content")
                if isinstance(c, str):
                    return c
            if delta is not None:
                c = getattr(delta, "content", None)
                if isinstance(c, str):
                    return c
            c = getattr(piece, "content", None)
            if isinstance(c, str):
                return c
            t = getattr(piece, "text", None)
            if isinstance(t, str):
                return t
            data = getattr(piece, "data", None)
            if isinstance(data, dict):
                c = data.get("content")
                if isinstance(c, str):
                    return c
            d = getattr(piece, "__dict__", None)
            if isinstance(d, dict):
                for k in ("content", "text"):
                    v = d.get(k)
                    if isinstance(v, str):
                        return v
            return ""

        answer = ""
        got_any_chunk = False

        # ❗️LẤY GENERATOR MỘT LẦN, bọc bằng _tap, rồi iterate trên biến cục bộ
        gen = response.response_gen
        gen = _tap(gen)

        for chunk in gen:
            got_any_chunk = True
            #print(f"[UI] got raw chunk: {chunk!r}")
            try:
                piece = _extract_text(chunk)
                #print(f"[UI] chunk type={type(chunk)} -> piece={piece!r}")
            except Exception as e:
                print(f"[UI] error while extracting chunk: {e!r}")
                piece = ""

            if not piece:
                continue

            answer += piece
            yield (
                DefaultElement.DEFAULT_MESSAGE,
                history + [[user_msg, answer]],
                DefaultElement.ANSWERING_STATUS,
            )

        #print("[UI] <<< EXIT stream_response")

        if not got_any_chunk and not answer:
            print("[UI] NO_CHUNKS_FROM_ENGINE (response_gen không nhả item nào)")

        yield (
            DefaultElement.DEFAULT_MESSAGE,
            history + [[user_msg, answer]],
            DefaultElement.COMPLETED_STATUS,
        )



class LocalChatbotUI:
    def __init__(
        self,
        pipeline: LocalRAGPipeline,
        logger: Logger,
        host: str = "host.docker.internal",
        data_dir: str = "data/data",
        avatar_images: list[str] = ["./assets/user.png", "./assets/bot.png"],
    ):
        self._pipeline = pipeline
        self._logger = logger
        self._host = host
        self._data_dir = os.path.join(os.getcwd(), data_dir)
        if not os.path.exists(self._data_dir):
            os.makedirs(self._data_dir, exist_ok=True)
        self._avatar_images = [
            os.path.join(os.getcwd(), image) for image in avatar_images
        ]
        self._variant = "panel"
        self._llm_response = LLMResponse()

    def _format_doc_list(self, paths: list[str]) -> str:
        if not paths:
            return "_(no documents)_"
        lines = [f"- {os.path.basename(p)}" for p in paths]
        return "\n".join(lines)

    
    def _get_respone(
        self,
        chat_mode: str,
        message: dict[str, str],
        chatbot: list[list[str, str]],
        progress=gr.Progress(track_tqdm=True),
    ):
        if self._pipeline.get_model_name() in [None, ""]:
            for m in self._llm_response.set_model():
                yield m
        elif message["text"] in [None, ""]:
            for m in self._llm_response.empty_message():
                yield m
        else:
            # console = sys.stdout
            # sys.stdout = self._logger
            response = self._pipeline.query(chat_mode, message["text"], chatbot)
            #print(f"[DEBUG] before stream_response: response.response_gen={response.response_gen!r}")
            for m in self._llm_response.stream_response(
                message, chatbot, response
            ):
                yield m
            # sys.stdout = console

    def _get_confirm_pull_model(self, model: str):
        if (model in ["gpt-3.5-turbo", "gpt-4"]) or (self._pipeline.check_exist(model)):
            self._change_model(model)
            return (
                gr.update(visible=False),
                gr.update(visible=False),
                DefaultElement.DEFAULT_STATUS,
            )
        return (
            gr.update(visible=True),
            gr.update(visible=True),
            DefaultElement.CONFIRM_PULL_MODEL_STATUS,
        )

    def _pull_model(self, model: str, progress=gr.Progress(track_tqdm=True)):
        if (model not in ["gpt-3.5-turbo", "gpt-4"]) and not (
            self._pipeline.check_exist(model)
        ):
            response = self._pipeline.pull_model(model)
            if response.status_code == 200:
                gr.Info(f"Pulling {model}!")
                for data in response.iter_lines(chunk_size=1):
                    data = json.loads(data)
                    if "completed" in data.keys() and "total" in data.keys():
                        progress(data["completed"] / data["total"], desc="Downloading")
                    else:
                        progress(0.0)
            else:
                gr.Warning(f"Model {model} doesn't exist!")
                return (
                    DefaultElement.DEFAULT_MESSAGE,
                    DefaultElement.DEFAULT_HISTORY,
                    DefaultElement.PULL_MODEL_FAIL_STATUS,
                    DefaultElement.DEFAULT_MODEL,
                )

        return (
            DefaultElement.DEFAULT_MESSAGE,
            DefaultElement.DEFAULT_HISTORY,
            DefaultElement.PULL_MODEL_SCUCCESS_STATUS,
            model,
        )

    def _change_model(self, model: str):
        if model not in [None, ""]:
            self._pipeline.set_model_name(model)
            self._pipeline.set_model()
            self._pipeline.set_engine()
            gr.Info(f"Change model to {model}!")
        return DefaultElement.DEFAULT_STATUS

    def _upload_document(self, document: list[str], list_files: list[str] | dict):
        if document in [None, []]:
            if isinstance(list_files, list):
                return (list_files, DefaultElement.DEFAULT_DOCUMENT)
            else:
                if list_files.get("files", None):
                    return list_files.get("files")
                return document
        else:
            if isinstance(list_files, list):
                return (document + list_files, DefaultElement.DEFAULT_DOCUMENT)
            else:
                if list_files.get("files", None):
                    return document + list_files.get("files")
                return document

    # def _reset_document(self):
    #     self._pipeline.reset_documents()
    #     gr.Info("Reset all documents!")
    #     return (
    #         DefaultElement.DEFAULT_DOCUMENT,
    #         gr.update(visible=False),
    #         gr.update(visible=False),
    #     )
    
    def _reset_document(self):
        self._pipeline.reset_documents()
        gr.Info("Reset all documents!")
        # Trả về: documents (Files) rỗng, markdown rỗng, 2 nút ẩn, state rỗng
        return (
            DefaultElement.DEFAULT_DOCUMENT,   # documents
            "_(no documents)_",                # docs_list_md
            gr.update(visible=False),          # upload_doc_btn
            gr.update(visible=False),          # reset_doc_btn
            [],                                # uploaded_files_state
        )


    # def _show_document_btn(self, document: list[str]):
    #     visible = False if document in [None, []] else True
    #     return (gr.update(visible=visible), gr.update(visible=visible))
    
    def _show_document_btn(self, uploaded_files_state: list[str]):
        visible = bool(uploaded_files_state)
        return (gr.update(visible=visible), gr.update(visible=visible))


    # def _processing_document(self, document: list[str], progress=gr.Progress(track_tqdm=True)):
    #     document = document or []
    #     input_files = []
    #     for file_path in document:
    #         dest = os.path.join(self._data_dir, os.path.basename(file_path))
    #         shutil.move(src=file_path, dst=dest)
    #         input_files.append(dest)
    #     self._pipeline.store_nodes(input_files=input_files)
    #     self._pipeline.set_chat_mode()
    #     gr.Info("Processing Completed!")
    #     return (self._pipeline.get_system_prompt(), DefaultElement.COMPLETED_STATUS)
    
    def _processing_document(
        self,
        document: list[str],
        uploaded_files_state: list[str] | None,
        progress=gr.Progress(track_tqdm=True),
    ):
        document = document or []
        uploaded_files_state = uploaded_files_state or []

        # 1) Move file vào thư mục data cố định
        input_files = []
        for file_path in document:
            dest = os.path.join(self._data_dir, os.path.basename(file_path))
            shutil.move(src=file_path, dst=dest)
            input_files.append(dest)

        # 2) ingest + chuyển engine sang chat mode
        if input_files:
            self._pipeline.store_nodes(input_files=input_files)
            self._pipeline.set_chat_mode()

        # 3) cập nhật state + markdown hiển thị
        new_uploaded = uploaded_files_state + input_files
        md = self._format_doc_list(new_uploaded)

        gr.Info("Processing Completed!")
        # Trả về: system_prompt, status, (reset Files về []), state mới, markdown hiển thị
        return (
            self._pipeline.get_system_prompt(),
            DefaultElement.COMPLETED_STATUS,
            gr.update(value=[]),        # ⚠️ rất quan trọng: xóa giá trị của Files sau ingest
            new_uploaded,
            md,
        )


    def _change_system_prompt(self, sys_prompt: str):
        self._pipeline.set_system_prompt(sys_prompt)
        self._pipeline.set_chat_mode()
        gr.Info("System prompt updated!")

    def _change_language(self, language: str):
        self._pipeline.set_language(language)
        self._pipeline.set_chat_mode()
        gr.Info(f"Change language to {language}")

    def _undo_chat(self, history: list[list[str, str]]):
        if len(history) > 0:
            history.pop(-1)
            return history
        return DefaultElement.DEFAULT_HISTORY

    def _reset_chat(self):
        self._pipeline.reset_conversation()
        gr.Info("Reset chat!")
        return (
            DefaultElement.DEFAULT_MESSAGE,
            DefaultElement.DEFAULT_HISTORY,
            DefaultElement.DEFAULT_DOCUMENT,
            DefaultElement.DEFAULT_STATUS,
        )

    def _clear_chat(self):
        self._pipeline.clear_conversation()
        gr.Info("Clear chat!")
        return (
            DefaultElement.DEFAULT_MESSAGE,
            DefaultElement.DEFAULT_HISTORY,
            DefaultElement.DEFAULT_STATUS,
        )

    def _show_hide_setting(self, state):
        state = not state
        label = "Hide Setting" if state else "Show Setting"
        return (label, gr.update(visible=state), state)

    def _welcome(self):
        for m in self._llm_response.welcome():
            yield m

    def build(self):
        with gr.Blocks(
            theme=gr.themes.Soft(primary_hue="slate"),
            js=JS_LIGHT_THEME,
            css=CSS,
        ) as demo:
            gr.Markdown("## Local RAG Chatbot 🤖")
            with gr.Tab("Interface"):
                sidebar_state = gr.State(True)
                with gr.Row(variant=self._variant, equal_height=False):
                    with gr.Column(
                        variant=self._variant, scale=10, visible=sidebar_state.value
                    ) as setting:
                        with gr.Column():
                            status = gr.Textbox(
                                label="Status", value="Ready!", interactive=False
                            )
                            language = gr.Radio(
                                label="Language",
                                choices=["vi", "eng"],
                                value="eng",
                                interactive=True,
                            )
                            model = gr.Text(
                                label="Model",
                                value=self._pipeline.get_model_name(),
                                interactive=False,
                            )
                            with gr.Row():
                                pull_btn = gr.Button(
                                    value="Pull Model", visible=False, min_width=50
                                )
                                cancel_btn = gr.Button(
                                    value="Cancel", visible=False, min_width=50
                                )

                            documents = gr.Files(
                                label="Add Documents",
                                value=[],
                                file_types=[".txt", ".pdf", ".csv"],
                                file_count="multiple",
                                height=150,
                                interactive=True,
                                
                            )
                            
                            uploaded_files_state = gr.State([])  # giữ danh sách file đã ingest
                            docs_list_md = gr.Markdown(label="Uploaded documents", value="_(no documents)_")

                            
                            with gr.Row():
                                upload_doc_btn = gr.UploadButton(
                                    label="Upload",
                                    value=[],
                                    file_types=[".txt", ".pdf", ".csv"],
                                    file_count="multiple",
                                    min_width=20,
                                    visible=False,
                                )
                                reset_doc_btn = gr.Button(
                                    "Reset", min_width=20, visible=False
                                )

                    with gr.Column(scale=30, variant=self._variant):
                        chatbot = gr.Chatbot(
                            layout="bubble",
                            value=[],
                            height=550,
                            scale=2,
                            show_copy_button=True,
                            bubble_full_width=False,
                            avatar_images=self._avatar_images,
                        )

                        with gr.Row(variant=self._variant):
                            chat_mode = gr.Dropdown(
                                choices=["chat", "QA"],
                                value="QA",
                                min_width=50,
                                show_label=False,
                                interactive=True,
                                allow_custom_value=False,
                            )
                            message = gr.MultimodalTextbox(
                                value=DefaultElement.DEFAULT_MESSAGE,
                                placeholder="Enter you message:",
                                file_types=[".txt", ".pdf", ".csv"],
                                show_label=False,
                                scale=6,
                                lines=1,
                            )
                        with gr.Row(variant=self._variant):
                            ui_btn = gr.Button(
                                value="Hide Setting"
                                if sidebar_state.value
                                else "Show Setting",
                                min_width=20,
                            )
                            undo_btn = gr.Button(value="Undo", min_width=20)
                            clear_btn = gr.Button(value="Clear", min_width=20)
                            reset_btn = gr.Button(value="Reset", min_width=20)

            with gr.Tab("Setting"):
                with gr.Row(variant=self._variant, equal_height=False):
                    with gr.Column():
                        system_prompt = gr.Textbox(
                            label="System Prompt",
                            value=self._pipeline.get_system_prompt(),
                            interactive=True,
                            lines=10,
                            max_lines=50,
                        )
                        sys_prompt_btn = gr.Button(value="Set System Prompt")

            with gr.Tab("Output"):
                with gr.Row(variant=self._variant):
                    log = gr.Code(
                        label="", language="markdown", interactive=False, lines=30
                    )
                    demo.load(
                        self._logger.read_logs,
                        outputs=[log],
                        every=1,
                        show_progress="hidden",
                        # scroll_to_output=True,
                    )

            clear_btn.click(self._clear_chat, outputs=[message, chatbot, status])
            cancel_btn.click(
                lambda: (gr.update(visible=False), gr.update(visible=False), None),
                outputs=[pull_btn, cancel_btn, model],
            )
            undo_btn.click(self._undo_chat, inputs=[chatbot], outputs=[chatbot])
            reset_btn.click(
                self._reset_chat, outputs=[message, chatbot, documents, status]
            )
            pull_btn.click(
                lambda: (gr.update(visible=False), gr.update(visible=False)),
                outputs=[pull_btn, cancel_btn],
            ).then(
                self._pull_model,
                inputs=[model],
                outputs=[message, chatbot, status, model],
            ).then(self._change_model, inputs=[model], outputs=[status])
            message.submit(
                self._upload_document, inputs=[documents, message], outputs=[documents]
            ).then(
                self._get_respone,
                inputs=[chat_mode, message, chatbot],
                outputs=[message, chatbot, status],
            )
            language.change(self._change_language, inputs=[language])
            model.change(
                self._get_confirm_pull_model,
                inputs=[model],
                outputs=[pull_btn, cancel_btn, status],
            )
            # documents.change(
            #     self._processing_document,
            #     inputs=[documents],
            #     outputs=[system_prompt, status],
            # ).then(
            #     self._show_document_btn,
            #     inputs=[documents],
            #     outputs=[upload_doc_btn, reset_doc_btn],
            # )
            documents.change(
                self._processing_document,
                inputs=[documents, uploaded_files_state],
                outputs=[system_prompt, status, documents, uploaded_files_state, docs_list_md],
            ).then(
                self._show_document_btn,
                inputs=[uploaded_files_state],
                outputs=[upload_doc_btn, reset_doc_btn],
            )


            sys_prompt_btn.click(self._change_system_prompt, inputs=[system_prompt])
            ui_btn.click(
                self._show_hide_setting,
                inputs=[sidebar_state],
                outputs=[ui_btn, setting, sidebar_state],
            )
            upload_doc_btn.upload(
                self._upload_document,
                inputs=[documents, upload_doc_btn],
                outputs=[documents, upload_doc_btn],
            )
            # reset_doc_btn.click(
            #     self._reset_document, outputs=[documents, upload_doc_btn, reset_doc_btn]
            # )
            reset_doc_btn.click(
                self._reset_document,
                outputs=[documents, docs_list_md, upload_doc_btn, reset_doc_btn, uploaded_files_state],
            )

            demo.load(self._welcome, outputs=[message, chatbot, status])

        return demo
