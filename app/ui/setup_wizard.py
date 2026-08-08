from __future__ import annotations

from PySide6.QtWidgets import (
    QWizard, QWizardPage, QLabel, QVBoxLayout, QTextEdit, QCheckBox, QLineEdit,
    QPushButton, QMessageBox,
)

from app.config import AppConfig
from app.hardware import detect_hardware
from app.model_registry import ModelRegistry


class WelcomePage(QWizardPage):
    def __init__(self):
        super().__init__()
        self.setTitle("Welcome to Friend")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "Let's set up your AI friend. We'll scan your computer, recommend models "
            "that fit it, and let you describe how your friend should talk. All of this "
            "runs locally by default -- nothing leaves your machine unless you turn on Gemini."
        ))


class HardwarePage(QWizardPage):
    def __init__(self):
        super().__init__()
        self.setTitle("Your computer")
        layout = QVBoxLayout(self)
        self.hw = detect_hardware()
        self.registry = ModelRegistry()
        recs = self.registry.recommend_all(self.hw)

        gpu_line = "Not detected (integrated/unknown graphics)"
        if self.hw.gpu.name:
            vram = f", ~{self.hw.gpu.vram_gb} GB VRAM ({self.hw.gpu.vram_confidence})" if self.hw.gpu.vram_gb else ""
            gpu_line = f"{self.hw.gpu.name}{vram}"

        text = (
            f"CPU: {self.hw.cpu_name}\n"
            f"Cores: {self.hw.cpu_cores_physical} physical / {self.hw.cpu_cores_logical} logical\n"
            f"RAM: {self.hw.ram_total_gb} GB total, {self.hw.ram_available_gb} GB available\n"
            f"GPU: {gpu_line}\n"
            f"Free disk space: {self.hw.disk_free_gb} GB\n"
            f"Performance profile: {self.hw.performance_tier}\n\n"
            "Recommended setup:\n"
        )
        for kind in ("asr", "llm", "tts"):
            rec = recs[kind]["recommended"]
            name = rec.raw["display_name"] if rec else "(nothing fits -- see disk/RAM above)"
            text += f"  {kind.upper()}: {name}\n"

        self.recommendations = recs
        label = QLabel(text)
        label.setStyleSheet("font-family: Consolas, monospace;")
        layout.addWidget(label)


class FriendPage(QWizardPage):
    def __init__(self):
        super().__init__()
        self.setTitle("Describe your friend")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "Write a short description of how your friend should talk and behave. "
            "You can fine-tune specific traits later in Settings."
        ))
        self.description_edit = QTextEdit()
        self.description_edit.setPlainText(
            "A friendly, casual person who talks naturally in Hindi and English, "
            "jokes sometimes, doesn't give unnecessarily long answers, and remembers "
            "things I tell them."
        )
        layout.addWidget(self.description_edit)


class GeminiPage(QWizardPage):
    def __init__(self):
        super().__init__()
        self.setTitle("Gemini (optional)")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "Friend works fully offline without this. Enabling Gemini lets it answer "
            "questions that need current information (news, weather) or the web. "
            "You can skip this and turn it on later."
        ))
        self.enable_checkbox = QCheckBox("Enable Gemini")
        layout.addWidget(self.enable_checkbox)

        self.key_input = QLineEdit()
        self.key_input.setPlaceholderText("Gemini API key (optional)")
        self.key_input.setEchoMode(QLineEdit.Password)
        layout.addWidget(self.key_input)

        self.model_input = QLineEdit("gemini-2.5-flash")
        self.model_input.setPlaceholderText("Gemini model identifier")
        layout.addWidget(self.model_input)


class SetupWizard(QWizard):
    def __init__(self, config: AppConfig):
        super().__init__()
        self.config = config
        self.setWindowTitle("Friend Setup")

        self.welcome_page = WelcomePage()
        self.hardware_page = HardwarePage()
        self.friend_page = FriendPage()
        self.gemini_page = GeminiPage()

        for page in (self.welcome_page, self.hardware_page, self.friend_page, self.gemini_page):
            self.addPage(page)

        self.finished.connect(self._apply_settings)

    def _apply_settings(self, result: int):
        if result != QWizard.Accepted:
            return

        recs = self.hardware_page.recommendations
        if recs["asr"]["recommended"]:
            self.config.asr_model_id = recs["asr"]["recommended"].id
        if recs["llm"]["recommended"]:
            self.config.llm_model_id = recs["llm"]["recommended"].id
        if recs["tts"]["recommended"]:
            self.config.tts_model_id = recs["tts"]["recommended"].id

        self.config.personality.description = self.friend_page.description_edit.toPlainText().strip()

        if self.gemini_page.enable_checkbox.isChecked():
            key = self.gemini_page.key_input.text().strip()
            if key:
                from app.providers.gemini_provider import save_api_key
                save_api_key(key)
            self.config.gemini_enabled = True
            self.config.gemini_model = self.gemini_page.model_input.text().strip() or "gemini-2.5-flash"
