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
    done = Signal(object)  # TurnResult
    failed = Signal(str)

    def __init__(self, pipeline: ConversationPipeline, text: str):
        super().__init__()
        self.pipeline = pipeline
        self.text = text

    def run(self):
        try:
            result = self.pipeline.run_turn_from_text(self.text)
            self.done.emit(result)
        except Exception as e:
            self.failed.emit(str(e))


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
        if not text:
            return
        self.text_input.clear()
        self._add_bubble(text, is_user=True)
        self.status_label.setText("Thinking...")
        self.turn_worker = TurnWorker(self.pipeline, text)
        self.turn_worker.done.connect(self._on_turn_done)
        self.turn_worker.failed.connect(self._on_turn_failed)
        self.turn_worker.start()

    def _on_turn_done(self, result):
        self._add_bubble(result.assistant_text, is_user=False)
        self.status_label.setText(f"Ready ({result.routed_to})")
        # Speak the reply. Kept synchronous-in-thread-free here for Phase 1 simplicity;
        # a follow-up pass should move this onto its own QThread so the UI never blocks
        # on audio playback, and wire Player.stop() to the mic button for barge-in.
        try:
            self.pipeline.speak(result.assistant_text, self.player)
        except ModelLoadError as e:
            self.status_label.setText(f"Voice output error: {e}")

    def _on_turn_failed(self, message: str):
        self.status_label.setText("Error")
        QMessageBox.warning(self, "Something went wrong", message)

    # --- voice turn (push-to-talk) ---
    def on_mic_pressed(self):
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
        try:
            result = self.pipeline.run_turn_from_audio(audio, sample_rate=16000)
        except ModelLoadError as e:
            QMessageBox.warning(self, "Speech recognition error", str(e))
            return
        if result.user_text:
            self._add_bubble(result.user_text, is_user=True)
            self._add_bubble(result.assistant_text, is_user=False)
            self.status_label.setText(f"Ready ({result.routed_to})")
            self.pipeline.speak(result.assistant_text, self.player)

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
        save_config(self.config)
        self.pipeline.unload_all()
        super().closeEvent(event)
