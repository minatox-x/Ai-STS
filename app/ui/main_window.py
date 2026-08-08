from __future__ import annotations

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QApplication, QLabel, QLineEdit, QMainWindow, QPushButton, QScrollArea,
    QVBoxLayout, QWidget, QHBoxLayout, QFrame, QMessageBox,
)

from app.audio_io import MicStream, Player
from app.config import AppConfig, save_config
from app.pipeline import ConversationPipeline
from app.providers.base import ModelLoadError


class ModelLoadWorker(QThread):
    status = Signal(str)
    finished_ok = Signal()
    failed = Signal(str, str)  # message, suggestion

    def __init__(self, pipeline: ConversationPipeline):
        super().__init__()
        self.pipeline = pipeline

    def run(self):
        try:
            self.pipeline.load_local_models(on_status=self.status.emit)
            try:
                self.pipeline.maybe_load_gemini()
            except ModelLoadError:
                pass  # Gemini is optional; local models are already loaded and usable.
            self.finished_ok.emit()
        except ModelLoadError as e:
            self.failed.emit(str(e), e.suggestion or "")
        except Exception as e:
            self.failed.emit(str(e), "")


class TurnWorker(QThread):
    """
    Runs one full conversation turn (ASR-if-needed -> LLM -> TTS/playback) off the
    main thread, for both the text path and the push-to-talk voice path.

    `turn_ready` fires as soon as the transcript/reply text is available, so the UI
    can update the chat bubbles immediately -- speaking then continues in this
    background thread (calling pipeline.speak() from here is safe: it only touches
    the audio device via sounddevice and never touches Qt widgets directly).
    """

    turn_ready = Signal(object)  # TurnResult
    speak_failed = Signal(str)
    failed = Signal(str)

    def __init__(
        self,
        pipeline: ConversationPipeline,
        player: Player,
        text: str | None = None,
        audio: bytes | None = None,
        sample_rate: int = 16000,
    ):
        super().__init__()
        self.pipeline = pipeline
        self.player = player
        self.text = text
        self.audio = audio
        self.sample_rate = sample_rate

    def run(self):
        try:
            if self.audio is not None:
                result = self.pipeline.run_turn_from_audio(self.audio, sample_rate=self.sample_rate)
            else:
                result = self.pipeline.run_turn_from_text(self.text or "")
        except ModelLoadError as e:
            self.failed.emit(str(e))
            return
        except Exception as e:
            self.failed.emit(str(e))
            return

        self.turn_ready.emit(result)

        if result.assistant_text:
            try:
                self.pipeline.speak(result.assistant_text, self.player)
            except ModelLoadError as e:
                self.speak_failed.emit(str(e))
            except Exception as e:
                self.speak_failed.emit(str(e))


class ChatBubble(QFrame):
    def __init__(self, text: str, is_user: bool):
        super().__init__()
        layout = QVBoxLayout(self)
        label = QLabel(text)
        label.setWordWrap(True)
        layout.addWidget(label)
        align = "right" if is_user else "left"
        bg = "#2b6cb0" if is_user else "#2d3748"
        self.setStyleSheet(
            f"QFrame {{ background-color: {bg}; border-radius: 10px; padding: 8px; }}"
            f"QLabel {{ color: white; }}"
        )
        self.setMaximumWidth(420)


class MainWindow(QMainWindow):
    def __init__(self, config: AppConfig):
        super().__init__()
        self.config = config
        self.pipeline = ConversationPipeline(config)
        self.player = Player()
        self.mic: MicStream | None = None
        self.is_recording = False

        self.setWindowTitle("Friend")
        self.resize(480, 640)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # header
        header = QHBoxLayout()
        title = QLabel("Friend")
        title.setStyleSheet("font-size: 20px; font-weight: 600;")
        header.addWidget(title)
        header.addStretch()
        self.status_label = QLabel("Loading...")
        self.status_label.setStyleSheet("color: #888;")
        header.addWidget(self.status_label)
        root.addLayout(header)

        # conversation area
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.chat_container = QWidget()
        self.chat_layout = QVBoxLayout(self.chat_container)
        self.chat_layout.addStretch()
        self.scroll.setWidget(self.chat_container)
        root.addWidget(self.scroll, stretch=1)

        # input row
        input_row = QHBoxLayout()
        self.text_input = QLineEdit()
        self.text_input.setPlaceholderText("Type a message...")
        self.text_input.returnPressed.connect(self.on_send_text)
        input_row.addWidget(self.text_input)

        self.mic_button = QPushButton("Hold to talk")
        self.mic_button.setEnabled(False)
        self.mic_button.pressed.connect(self.on_mic_pressed)
        self.mic_button.released.connect(self.on_mic_released)
        input_row.addWidget(self.mic_button)

        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.on_send_text)
        input_row.addWidget(self.send_button)
        root.addLayout(input_row)

        self._start_model_loading()

    # --- model loading ---
    def _start_model_loading(self):
        self.load_worker = ModelLoadWorker(self.pipeline)
        self.load_worker.status.connect(self.status_label.setText)
        self.load_worker.finished_ok.connect(self._on_models_ready)
        self.load_worker.failed.connect(self._on_models_failed)
        self.load_worker.start()

    def _on_models_ready(self):
        self.status_label.setText("Ready")
        self.mic_button.setEnabled(True)
        self.send_button.setEnabled(True)

    def _on_models_failed(self, message: str, suggestion: str):
        self.status_label.setText("Error loading models")
        text = message + (f"\n\n{suggestion}" if suggestion else "")
        QMessageBox.critical(self, "Couldn't load models", text)

    # --- text turn ---
    def on_send_text(self):
        text = self.text_input.text().strip()
        if not text or self._turn_in_progress():
            return
        self.text_input.clear()
        self._add_bubble(text, is_user=True)
        self.status_label.setText("Thinking...")
        self._set_inputs_busy(True)
        self.turn_worker = TurnWorker(self.pipeline, self.player, text=text)
        self.turn_worker.turn_ready.connect(self._on_turn_ready)
        self.turn_worker.speak_failed.connect(self._on_speak_failed)
        self.turn_worker.failed.connect(self._on_turn_failed)
        self.turn_worker.finished.connect(self._on_worker_finished)
        self.turn_worker.start()

    def _on_turn_ready(self, result):
        # Fired as soon as the reply text is ready -- speaking happens after this,
        # still inside the worker thread, so the UI never blocks on it.
        self._add_bubble(result.assistant_text, is_user=False)
        self.status_label.setText(f"Speaking... ({result.routed_to})")

    def _on_speak_failed(self, message: str):
        self.status_label.setText(f"Voice output error: {message}")

    def _on_turn_failed(self, message: str):
        self.status_label.setText("Error")
        QMessageBox.warning(self, "Something went wrong", message)

    def _on_worker_finished(self):
        self._set_inputs_busy(False)
        if not self.status_label.text().startswith(("Error", "Voice output error")):
            self.status_label.setText("Ready")

    # --- voice turn (push-to-talk) ---
    def on_mic_pressed(self):
        if self._turn_in_progress():
            return
        self.mic = MicStream()
        self.mic.start()
        self.is_recording = True
        self.mic_button.setText("Listening...")

    def on_mic_released(self):
        if not self.mic:
            return
        audio = self.mic.stop()
        self.is_recording = False
        self.mic_button.setText("Hold to talk")
        if not audio:
            return
        self.status_label.setText("Transcribing...")
        self._set_inputs_busy(True)
        # Run ASR -> LLM -> TTS entirely off the main thread so the window never
        # freezes while the model is thinking / speaking (this used to call
        # run_turn_from_audio() and speak() directly here, on the UI thread).
        self.turn_worker = TurnWorker(self.pipeline, self.player, audio=audio, sample_rate=16000)
        self.turn_worker.turn_ready.connect(self._on_voice_turn_ready)
        self.turn_worker.speak_failed.connect(self._on_speak_failed)
        self.turn_worker.failed.connect(self._on_voice_turn_failed)
        self.turn_worker.finished.connect(self._on_worker_finished)
        self.turn_worker.start()

    def _on_voice_turn_ready(self, result):
        if not result.user_text:
            self.status_label.setText("Didn't catch that -- try again.")
            return
        self._add_bubble(result.user_text, is_user=True)
        self._add_bubble(result.assistant_text, is_user=False)
        self.status_label.setText(f"Speaking... ({result.routed_to})")

    def _on_voice_turn_failed(self, message: str):
        self.status_label.setText("Error")
        QMessageBox.warning(self, "Speech recognition error", message)

    # --- shared busy-state helpers ---
    def _turn_in_progress(self) -> bool:
        worker = getattr(self, "turn_worker", None)
        return worker is not None and worker.isRunning()

    def _set_inputs_busy(self, busy: bool):
        self.mic_button.setEnabled(not busy)
        self.send_button.setEnabled(not busy)
        self.text_input.setEnabled(not busy)

    def _add_bubble(self, text: str, is_user: bool):
        bubble = ChatBubble(text, is_user)
        row = QHBoxLayout()
        if is_user:
            row.addStretch()
            row.addWidget(bubble)
        else:
            row.addWidget(bubble)
            row.addStretch()
        wrapper = QWidget()
        wrapper.setLayout(row)
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, wrapper)

    def closeEvent(self, event):
        # Stop any in-flight playback/turn before tearing down the models, otherwise
        # a background TurnWorker can be mid-generation against providers that
        # unload_all() is about to null out from under it.
        self.player.stop()
        worker = getattr(self, "turn_worker", None)
        if worker is not None and worker.isRunning():
            worker.wait(2000)
        save_config(self.config)
        self.pipeline.unload_all()
        super().closeEvent(event)
