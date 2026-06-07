"""
nixorb/main.py — NixOrb daemon entry point.

Threading model
───────────────
  Main thread  → Qt GUI via qasync QEventLoop
  asyncio loop → EventBus, LLM streaming, TTS, web search
  Thread pool  → sounddevice recording, Whisper, VRAM I/O

QSocketNotifier / QThread warning fix
──────────────────────────────────────
  The warning fires when asyncio tries to install a Unix-socket-based
  SIGINT watcher inside Qt's event loop. Fix: catch KeyboardInterrupt
  at the outermost level and use app.aboutToQuit to trigger clean
  shutdown — no loop.add_signal_handler(), no signal.signal().
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import sys
from typing import Any

log = logging.getLogger(__name__)

_SCREEN_KW = frozenset({
    "screen", "looking at", "what's on", "what is on",
    "see my screen", "my display", "show me my",
})
_WEB_KW = frozenset({
    "search", "look up", "google", "find out", "what is",
    "who is", "when did", "latest", "news", "current",
    "today", "right now", "recently",
})

SYSTEM_PROMPT = """\
You are NixOrb — a capable AI assistant running inside Arch Linux, \
voiced with GLaDOS-style dry wit. You live as a glowing orb on the \
user's Wayland desktop.

Personality: precise, occasionally sardonic, never rude. You know \
Arch Linux deeply. Keep responses concise unless depth is requested.

Capabilities you have RIGHT NOW:
1. TERMINAL — wrap bash in <ACTION>command</ACTION>.
   Only when asked or when a task clearly requires it.
   Always briefly explain what the command does.
2. WEB SEARCH — auto-injected when your query seems to need it.
3. SCREEN — you can see the desktop when asked.
4. MEMORY — past conversations retrieved via vector search.
5. PLUGINS — user-installed tools available as function calls.

Rules:
- Never use <ACTION> for explanations — only real executable commands.
- Warn before destructive operations.
- Use fenced code blocks for code.
- If unsure about current facts, say so rather than hallucinate.

System: Arch Linux · KDE Plasma 6 · Wayland · NVIDIA GTX 1080 · Python 3.12"""


def _strip_actions(text: str) -> str:
    return re.sub(r"<ACTION>.*?</ACTION>", "", text, flags=re.DOTALL).strip()

def _wants_screen(text: str) -> bool:
    return any(kw in text.lower() for kw in _SCREEN_KW)

def _wants_web(text: str) -> bool:
    return any(kw in text.lower() for kw in _WEB_KW)


def _build_llm(settings):
    from nixorb.llm.backends import (
        HuggingFaceBackend,
        LocalLLMBackend,
        OfflineFallbackManager,
        OllamaBackend,
        OpenAIBackend,
    )
    b = settings.llm_backend.lower()
    if b == "openai":
        primary = OpenAIBackend(
            settings.openai_api_key, settings.llm_model, settings.llm_base_url
        )
    elif b == "ollama":
        primary = OllamaBackend(settings.llm_model)
    elif b == "local":
        primary = LocalLLMBackend(settings.local_model_path, settings.llm_vram_mb)
    else:
        primary = HuggingFaceBackend(
            settings.llm_model, settings.hf_token,
            settings.llm_vram_mb, settings.llm_max_new_tokens,
        )
    if settings.offline_fallback_enabled and settings.fallback_model_path:
        fallback = LocalLLMBackend(settings.fallback_model_path, 2048)
        return OfflineFallbackManager(primary, fallback)
    return primary


async def _async_main(settings, app) -> None:
    from nixorb.core.aur_checker import check_dependencies
    from nixorb.core.event_bus import Event, bus
    from nixorb.core.vram_manager import vram
    from nixorb.utils.logger import setup_logging

    setup_logging(log_to_file=True)

    await bus.start()
    log.info("NixOrb %s starting", settings.__class__.__module__)

    # Show Qt surfaces before dependency/model initialisation so packaged
    # launches never look hung while models are being cached.
    from nixorb.ui.settings_window import SettingsWindow
    SettingsWindow.init_settings(settings)

    from nixorb.ui.tray_icon import NixOrbTray
    tray = NixOrbTray(settings, app)
    tray.show()

    from nixorb.ui.orb_window import OrbWindow
    orb = OrbWindow(settings, app)
    orb.show()

    await vram.start_monitor(poll_interval=6.0)

    for pkg in check_dependencies():
        await bus.emit(
            Event.LOG,
            data={"level": "warning", "msg": f"⚠  Missing package: {pkg}"},
            source="startup",
        )

    from nixorb.memory.vector_store import VectorMemory
    memory = VectorMemory(settings.memory_dir)

    from nixorb.asr.whisper_engine import WhisperEngine
    asr = WhisperEngine(settings)

    from nixorb.vision.screen_capture import ScreenCapture
    screen = ScreenCapture()

    llm = _build_llm(settings)

    from nixorb.tts.tts_factory import build_tts
    tts = build_tts(settings)

    from nixorb.action.executor import ActionExecutor
    executor = ActionExecutor(settings)

    from nixorb.plugins.loader import PluginLoader
    plugin_loader = PluginLoader(settings.plugin_dir)
    plugin_loader.load_all()
    names = plugin_loader.plugin_names()
    log.info("Plugins loaded: %s", ", ".join(names) if names else "none")

    async def _preload_asr() -> None:
        try:
            await asr.preload()
            await bus.emit(Event.LOG, data={"level": "info", "msg": "✅ ASR model ready"}, source="startup")
        except Exception as exc:
            log.warning("ASR model preload failed: %s", exc)
            await bus.emit(Event.LOG, data={"level": "warning", "msg": f"⚠ ASR preload failed: {exc}"}, source="startup")

    asyncio.create_task(_preload_asr(), name="nixorb-asr-preload")

    # ── Hotkey (after bus._loop confirmed set) ────────────────────── #
    from nixorb.ui.hotkey import HotkeyManager
    HotkeyManager(settings).start()

    # ── Wake word ─────────────────────────────────────────────────── #
    if settings.wake_word_enabled:
        from nixorb.asr.wake_word import WakeWordDetector
        asyncio.create_task(
            WakeWordDetector(settings).run_forever(), name="wake-word"
        )

    _mic_muted = False

    async def _on_mic_muted(payload) -> None:
        nonlocal _mic_muted
        _mic_muted = bool((payload.data or {}).get("muted", False))
        log.info("Mic %s", "muted" if _mic_muted else "unmuted")

    bus.subscribe(Event.MIC_MUTED, _on_mic_muted)

    conversation: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    async def _handle_turn(_payload) -> None:
        nonlocal _mic_muted
        if _mic_muted:
            log.debug("Mic muted — ignoring trigger")
            return

        await bus.emit(Event.ORB_LISTENING, source="main")
        log.info("Listening…")

        transcript = await asr.record_and_transcribe()
        if not transcript:
            log.info("No speech detected")
            await bus.emit(Event.ORB_IDLE, source="main")
            return

        log.info("Transcript: %s", transcript)
        await bus.emit(
            Event.LOG,
            data={"level": "info", "msg": f"🎙 You: {transcript}"},
            source="main",
        )

        mem_ctx  = memory.build_context_block(transcript)
        user_msg = (mem_ctx + transcript) if mem_ctx else transcript

        if settings.clipboard_enabled and "clipboard" in transcript.lower():
            from nixorb.action.clipboard import read_clipboard
            clip = await read_clipboard()
            if clip:
                user_msg += f"\n\n[Clipboard]:\n{clip}"
                log.debug("Clipboard injected (%d chars)", len(clip))

        if settings.web_search_enabled and _wants_web(transcript):
            from nixorb.utils.web_search import search_formatted
            log.info("Web search: %s", transcript[:60])
            web_ctx = await search_formatted(transcript, settings.web_search_max_results)
            user_msg += f"\n\n{web_ctx}"

        if settings.screen_capture_enabled and _wants_screen(transcript):
            await bus.emit(Event.SCREEN_CAPTURE_REQ, source="main")
            log.info("Screen capture requested")
            if settings.use_vlm:
                desc = await screen.describe(llm, question=transcript)
            else:
                desc = await screen.describe_cogflorence(
                    settings.vision_model, settings.hf_token
                )
            user_msg += f"\n\n[Screen]: {desc}"
            await bus.emit(Event.SCREEN_CAPTURE_DONE, source="main")

        conversation.append({"role": "user", "content": user_msg})
        await vram.evict("whisper")
        await bus.emit(Event.ORB_THINKING, source="main")
        log.info("Querying LLM: %s", settings.llm_model)

        full: list[str] = []
        try:
            tools = plugin_loader.get_tool_definitions() or None
            async for chunk in llm.stream(conversation, tools=tools):
                full.append(chunk)
        except Exception as exc:
            log.error("LLM error: %s", exc)
            await bus.emit(Event.LLM_ERROR, data={"error": str(exc)}, source="main")
            await bus.emit(Event.ORB_ERROR, source="main")
            await bus.emit(
                Event.LOG,
                data={"level": "error", "msg": f"❌ LLM error: {exc}"},
                source="main",
            )
            await asyncio.sleep(2)
            await bus.emit(Event.ORB_IDLE, source="main")
            return

        response = "".join(full)
        conversation.append({"role": "assistant", "content": response})
        log.info("Response (%d chars): %s", len(response), response[:100])
        await bus.emit(
            Event.LOG,
            data={"level": "info", "msg": f"🤖 NixOrb: {response[:200]}"},
            source="main",
        )

        memory.store(
            f"User: {transcript}\nAssistant: {response[:600]}",
            metadata={"type": "conversation"},
        )

        results = await executor.handle_llm_output(response)
        if results:
            result_text = "\n\n".join(str(r) for r in results)
            conversation.append(
                {"role": "user", "content": f"<RESULT>\n{result_text}\n</RESULT>"}
            )
            if any(r.stdout for r in results):
                followup: list[str] = []
                with contextlib.suppress(Exception):
                    async for chunk in llm.stream(conversation):
                        followup.append(chunk)
                if followup:
                    ftext = "".join(followup)
                    conversation.append({"role": "assistant", "content": ftext})
                    response = ftext

        if settings.clipboard_enabled:
            from nixorb.action.clipboard import write_clipboard
            code_blocks = re.findall(r"```(?:\w+)?\n(.*?)```", response, re.DOTALL)
            if code_blocks:
                await write_clipboard(code_blocks[-1].strip())
                log.debug("Copied code block to clipboard")

        await bus.emit(Event.ORB_SPEAKING, source="main")
        speech    = _strip_actions(response)
        sentences = re.split(r"(?<=[.!?])\s+", speech)
        tts_text  = " ".join(sentences[:6]) if len(sentences) > 6 else speech
        if tts_text:
            log.info("Speaking: %s", tts_text[:80])
            await tts.speak(tts_text)

        await bus.emit(Event.ORB_IDLE, source="main")

        if len(conversation) > 22:
            conversation[1:] = conversation[-20:]

    bus.subscribe(Event.HOTKEY_TRIGGERED,   _handle_turn, priority=2)
    bus.subscribe(Event.WAKE_WORD_DETECTED, _handle_turn, priority=2)

    async def _log_to_python(payload) -> None:
        data  = payload.data or {}
        level = data.get("level", "info")
        msg   = data.get("msg", "")
        getattr(
            log, level if level in ("debug", "info", "warning", "error") else "info"
        )("[bus] %s", msg)

    bus.subscribe(Event.LOG, _log_to_python)

    # ── Shutdown via app.aboutToQuit (no signal handlers = no QSocketNotifier) ─ #
    stop_event = asyncio.Event()

    def _on_qt_quit() -> None:
        log.info("Qt quit signal received")
        stop_event.set()

    app.aboutToQuit.connect(_on_qt_quit)

    async def _on_bus_shutdown(_payload) -> None:
        stop_event.set()

    bus.subscribe(Event.SHUTDOWN, _on_bus_shutdown)

    from nixorb import __version__
    log.info("NixOrb %s ready — hotkey: %s  LLM: %s", __version__, settings.hotkey, settings.llm_model)
    await bus.emit(
        Event.LOG,
        data={"level": "success",
              "msg": f"✅ NixOrb {__version__} ready | hotkey: {settings.hotkey} | LLM: {settings.llm_model}"},
        source="startup",
    )

    await stop_event.wait()
    log.info("Shutting down…")
    await vram.stop()
    await bus.stop()


def main() -> None:
    import qasync
    from PySide6.QtWidgets import QApplication

    from nixorb.settings import Settings
    settings = Settings.load()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    app = QApplication.instance() or QApplication(sys.argv)
    from nixorb import __version__
    app.setApplicationName("NixOrb")
    app.setApplicationVersion(__version__)
    app.setOrganizationName("NixOrb")
    app.setQuitOnLastWindowClosed(False)

    # QEventLoop from qasync — wraps Qt event loop with asyncio
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
