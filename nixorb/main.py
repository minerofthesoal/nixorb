"""NixOrb main entry point — AI assistant daemon.

Architecture:
  Qt Main Thread     asyncio Event Loop        Thread Pool
  ───────────────    ──────────────────        ───────────
  OrbWindow (QML) ←  EventBus                 Whisper inference
  SettingsWindow  ←  LLM streaming             Piper TTS
  NixOrbTray      ←  VRAMManager               Command execution
  HotkeyManager      PluginLoader
                     VectorMemory (ChromaDB)

Pipeline:
  Hotkey/WakeWord → Record Audio → Whisper STT → Ollama LLM →
  Piper TTS → Speak + Execute Actions
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import sys

log = logging.getLogger(__name__)

# Keywords for feature detection
_SCREEN_KW = frozenset({
    "screen", "looking at", "what's on", "what is on",
    "see my screen", "my display", "show me my",
})
_WEB_KW = frozenset({
    "search", "look up", "google", "find out", "what is",
    "who is", "when did", "latest", "news", "current",
    "today", "right now", "recently",
})


def _strip_actions(text: str) -> str:
    """Remove <ACTION> tags from text for TTS."""
    return re.sub(r"<ACTION>.*?</ACTION>", "", text, flags=re.DOTALL).strip()


def _wants_screen(text: str) -> bool:
    """Check if the user is asking about their screen."""
    return any(kw in text.lower() for kw in _SCREEN_KW)


def _wants_web(text: str) -> bool:
    """Check if the user wants a web search."""
    return any(kw in text.lower() for kw in _WEB_KW)


def _disable_crashing_accessibility_bridge() -> None:
    """Prevent KDE Plasma AT-SPI accessibility bridge crash.

    On KDE Plasma sessions, Qt auto-constructs a QSpiAccessibleBridge
    which crashes inside PySide6. Setting QT_ACCESSIBILITY=0 prevents this.
    """
    os.environ.setdefault("QT_ACCESSIBILITY", "0")


def _select_qt_platform() -> None:
    """Select a working Qt platform plugin.

    On Wayland sessions without a native Qt Wayland plugin, force xcb
    (XWayland) which is more reliable with pip-installed PySide6.
    """
    if os.environ.get("QT_QPA_PLATFORM"):
        return

    session_type = os.environ.get("XDG_SESSION_TYPE", "").lower()
    has_wayland = bool(os.environ.get("WAYLAND_DISPLAY"))
    has_x11 = bool(os.environ.get("DISPLAY"))

    if session_type == "wayland" or has_wayland:
        if has_x11:
            os.environ["QT_QPA_PLATFORM"] = "xcb"
            log.info("Qt: Wayland session → using xcb via XWayland")
        else:
            os.environ.setdefault("QT_QPA_PLATFORM", "wayland")
            log.warning(
                "Qt: No XWayland — attempting native wayland plugin. "
                "Install qt6-wayland if this fails."
            )
    elif not has_x11:
        log.error(
            "Qt: No display detected — NixOrb needs a graphical session."
        )


async def _async_main(settings, app) -> None:
    """Main async orchestrator — initializes all components."""
    from PySide6.QtWidgets import QSystemTrayIcon

    from nixorb.core.event_bus import Event, bus
    from nixorb.core.vram_manager import vram
    from nixorb.memory.vector_store import VectorMemory
    from nixorb.ui.orb_window import OrbWindow
    from nixorb.ui.settings_window import SettingsWindow
    from nixorb.ui.tray_icon import NixOrbTray
    from nixorb.utils.logger import setup_logging

    setup_logging(log_to_file=True)

    await bus.start()
    log.info("NixOrb %s starting", __import__("nixorb").__version__)

    # Prime qasync cross-thread wakeup from Qt thread
    # This prevents "QSocketNotifier: Can only be used with threads started with QThread"
    asyncio.get_running_loop().call_soon_threadsafe(lambda: None)

    # ── Initialize UI ────────────────────────────────────────────── #
    SettingsWindow.init_settings = lambda s: None  # type: ignore

    if QSystemTrayIcon.isSystemTrayAvailable():
        tray = NixOrbTray(settings, app)
        tray.show()
        log.info("Tray: system tray icon active")
    else:
        log.warning("Tray: system tray not available")

    orb = OrbWindow(settings, app)
    orb.show()
    orb.log_visibility()

    # ── Initialize core services ─────────────────────────────────── #
    await vram.start_monitor(poll_interval=6.0)

    # Memory
    memory = VectorMemory(settings.memory_dir)

    # ASR (Whisper)
    from nixorb.asr.whisper_engine import WhisperEngine
    asr = WhisperEngine(settings)

    # LLM (Ollama — local only)
    from nixorb.llm.ollama_backend import OllamaBackend
    llm = OllamaBackend(settings)

    # Check Ollama health
    health = await llm.health_check()
    if health["ok"]:
        log.info("LLM: Ollama ready with model '%s'", settings.llm_model)
    else:
        log.warning("LLM: %s", health.get("error", "Unknown error"))
        log.info("LLM: Run 'ollama pull %s' to download the model", settings.llm_model)

    # TTS (Piper)
    from nixorb.tts.piper_tts import PiperTTS
    tts = PiperTTS(settings)

    # Action executor
    from nixorb.action.executor import ActionExecutor
    executor = ActionExecutor(settings)

    # Confirmation dialog handler
    from nixorb.ui.confirm_dialog import register_confirmation_handler
    register_confirmation_handler()

    # Plugin loader
    from nixorb.plugins.loader import PluginLoader
    plugin_loader = PluginLoader(settings.plugin_dir)
    if settings.plugins_enabled:
        plugin_loader.load_all()

    # ── Preload ASR model ────────────────────────────────────────── #
    async def _preload_asr() -> None:
        try:
            await asr.preload()
            await bus.emit(
                Event.LOG,
                data={"level": "info", "msg": "✅ ASR model ready"},
                source="startup",
            )
        except Exception as exc:
            log.warning("ASR preload failed: %s", exc)
            await bus.emit(
                Event.LOG,
                data={"level": "warning", "msg": f"⚠ ASR preload failed: {exc}"},
                source="startup",
            )

    asyncio.create_task(_preload_asr(), name="asr-preload")

    # ── Hotkey manager ───────────────────────────────────────────── #
    from nixorb.ui.hotkey import HotkeyManager
    HotkeyManager(settings).start()

    # ── Wake word detector ───────────────────────────────────────── #
    wake_word = None
    if settings.wake_word_enabled:
        from nixorb.asr.wake_word import WakeWordDetector
        wake_word = WakeWordDetector(settings)
        asyncio.create_task(wake_word.run_forever(), name="wake-word")

    # ── Mic mute state ───────────────────────────────────────────── #
    mic_muted = False

    async def _on_mic_muted(payload) -> None:
        nonlocal mic_muted
        mic_muted = bool((payload.data or {}).get("muted", False))
        log.info("Mic %s", "muted" if mic_muted else "unmuted")

    bus.subscribe(Event.MIC_MUTED, _on_mic_muted)

    # ── Main conversation handler ────────────────────────────────── #
    conversation: list[dict[str, str]] = [
        {"role": "system", "content": settings.llm_system_prompt}
    ]

    async def _handle_turn(payload) -> None:
        """Handle a single conversation turn."""
        nonlocal mic_muted

        if mic_muted:
            log.debug("Mic muted — ignoring trigger")
            return

        await bus.emit(Event.ORB_LISTENING, source="main")
        log.info("🎙 Listening…")

        # Record and transcribe
        transcript = await asr.record_and_transcribe()
        if not transcript:
            log.info("No speech detected")
            await bus.emit(Event.ORB_IDLE, source="main")
            return

        log.info("📝 Transcript: %s", transcript)
        await bus.emit(
            Event.LOG,
            data={"level": "info", "msg": f"🎙 You: {transcript}"},
            source="main",
        )

        # Build user message with context
        user_msg = transcript

        # Add memory context
        if settings.memory_enabled:
            mem_ctx = memory.build_context_block(transcript)
            if mem_ctx:
                user_msg = mem_ctx + transcript

        # Check clipboard
        if settings.clipboard_enabled and "clipboard" in transcript.lower():
            from nixorb.action.clipboard import read_clipboard
            clip = await read_clipboard()
            if clip:
                user_msg += f"\n\n[Clipboard]:\n{clip}"
                log.debug("Clipboard injected (%d chars)", len(clip))

        # Check web search
        if settings.web_search_enabled and _wants_web(transcript):
            try:
                from nixorb.utils.web_search import search_formatted
                log.info("🔍 Web search: %s", transcript[:60])
                web_ctx = await search_formatted(transcript, settings.web_search_max_results)
                user_msg += f"\n\n{web_ctx}"
            except Exception as exc:
                log.warning("Web search failed: %s", exc)

        # Check screen capture
        if settings.screen_capture_enabled and _wants_screen(transcript):
            await bus.emit(Event.SCREEN_CAPTURE_REQ, source="main")
            try:
                from nixorb.vision.screen_capture import ScreenCapture
                screen = ScreenCapture()
                desc = await screen.describe()
                user_msg += f"\n\n[Screen]: {desc}"
                await bus.emit(Event.SCREEN_CAPTURE_DONE, source="main")
            except Exception as exc:
                log.warning("Screen capture failed: %s", exc)
                await bus.emit(Event.SCREEN_CAPTURE_DONE, source="main")

        # Add to conversation
        conversation.append({"role": "user", "content": user_msg})

        # Unload Whisper to free VRAM for LLM
        await asr.unload()
        await bus.emit(Event.ORB_THINKING, source="main")
        log.info("🤔 Querying LLM: %s", settings.llm_model)

        # Stream LLM response
        full_response: list[str] = []
        try:
            tools = plugin_loader.get_tool_definitions() or None
            async for chunk in llm.stream(conversation, tools=tools):
                full_response.append(chunk)

            response = "".join(full_response)

        except Exception as exc:
            log.error("LLM error: %s", exc)
            await bus.emit(Event.LLM_ERROR, data={"error": str(exc)}, source="main")
            await bus.emit(Event.ORB_ERROR, source="main")
            await asyncio.sleep(2)
            await bus.emit(Event.ORB_IDLE, source="main")
            return

        # Add response to conversation
        conversation.append({"role": "assistant", "content": response})
        log.info("🤖 Response (%d chars): %s", len(response), response[:100])
        await bus.emit(
            Event.LOG,
            data={"level": "info", "msg": f"🤖 NixOrb: {response[:200]}"},
            source="main",
        )

        # Store in memory
        if settings.memory_enabled:
            memory.store(
                f"User: {transcript}\nAssistant: {response[:600]}",
                metadata={"type": "conversation"},
            )

        # Execute any actions
        action_results = await executor.handle_llm_output(response)
        if action_results:
            result_texts = []
            for r in action_results:
                if r.stdout:
                    result_texts.append(r.stdout)
            if result_texts:
                result_msg = "\n\n".join(result_texts)
                conversation.append(
                    {"role": "user", "content": f"<RESULT>\n{result_msg}\n</RESULT>"}
                )
                # Get follow-up response
                followup_chunks: list[str] = []
                try:
                    async for chunk in llm.stream(conversation):
                        followup_chunks.append(chunk)
                    if followup_chunks:
                        followup = "".join(followup_chunks)
                        conversation.append({"role": "assistant", "content": followup})
                        response = followup
                except Exception as exc:
                    log.warning("Follow-up LLM call failed: %s", exc)

        # Copy code blocks to clipboard
        if settings.clipboard_enabled:
            from nixorb.action.clipboard import write_clipboard
            code_blocks = re.findall(r"```(?:\w+)?\n(.*?)```", response, re.DOTALL)
            if code_blocks:
                await write_clipboard(code_blocks[-1].strip())
                log.debug("Copied code block to clipboard")

        # Speak response
        await bus.emit(Event.ORB_SPEAKING, source="main")
        speech_text = _strip_actions(response)
        # Limit to first 6 sentences for TTS
        sentences = re.split(r"(?<=[.!?])\s+", speech_text)
        tts_text = " ".join(sentences[:6]) if len(sentences) > 6 else speech_text

        if tts_text:
            log.info("🔊 Speaking: %s", tts_text[:80])
            await tts.speak(tts_text)

        await bus.emit(Event.ORB_IDLE, source="main")

        # Trim conversation history
        if len(conversation) > 22:
            conversation[1:] = conversation[-20:]

    # Subscribe to triggers
    bus.subscribe(Event.HOTKEY_TRIGGERED, _handle_turn, priority=2)
    bus.subscribe(Event.WAKE_WORD_DETECTED, _handle_turn, priority=2)
    bus.subscribe(Event.ORB_CLICKED, _handle_turn, priority=2)

    # Log handler
    async def _log_to_python(payload) -> None:
        data = payload.data or {}
        level = data.get("level", "info")
        msg = data.get("msg", "")
        getattr(
            log,
            level if level in ("debug", "info", "warning", "error") else "info",
        )("[bus] %s", msg)

    bus.subscribe(Event.LOG, _log_to_python)

    # ── Shutdown handling ────────────────────────────────────────── #
    stop_event = asyncio.Event()

    def _on_qt_quit() -> None:
        log.info("Qt quit signal received")
        stop_event.set()

    app.aboutToQuit.connect(_on_qt_quit)

    async def _on_shutdown(_payload) -> None:
        stop_event.set()

    bus.subscribe(Event.SHUTDOWN, _on_shutdown)

    # ── Ready ────────────────────────────────────────────────────── #
    log.info(
        "✅ NixOrb %s ready — hotkey: %s | LLM: %s | model: %s",
        __import__("nixorb").__version__,
        settings.hotkey,
        settings.llm_backend,
        settings.llm_model,
    )
    await bus.emit(
        Event.LOG,
        data={
            "level": "success",
            "msg": (
                f"✅ NixOrb ready | hotkey: {settings.hotkey} "
                f"| LLM: {settings.llm_model}"
            ),
        },
        source="startup",
    )

    # Wait for shutdown
    await stop_event.wait()
    log.info("Shutting down…")

    # Cleanup
    if wake_word:
        wake_word.stop()
    await asr.unload()
    await llm.close()
    await vram.stop()
    await bus.stop()


def main() -> None:
    """Entry point — initializes Qt and starts the async loop."""
    import qasync
    from PySide6.QtWidgets import QApplication

    from nixorb.settings import Settings

    settings = Settings.load()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    # Prevent Qt crashes on KDE
    _disable_crashing_accessibility_bridge()
    _select_qt_platform()

    # Create Qt application
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("NixOrb")
    app.setApplicationVersion(__import__("nixorb").__version__)
    app.setOrganizationName("NixOrb")
    app._quit_on_last_window_closed = False  # type: ignore[attr-defined]
    app.setQuitOnLastWindowClosed(False)

    # qasync: integrate asyncio with Qt event loop
    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)

    with loop:
        try:
            loop.run_until_complete(_async_main(settings, app))
        except KeyboardInterrupt:
            log.info("KeyboardInterrupt — shutting down")
        finally:
            loop.close()


if __name__ == "__main__":
    main()
