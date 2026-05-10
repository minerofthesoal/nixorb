"""
nixorb/main.py — NixOrb daemon.

QSocketNotifier fix: use signal.signal() + asyncio.sleep() poll instead of
loop.add_signal_handler() which creates a Unix socketpair Qt complains about.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import signal
import sys
from typing import Any

log = logging.getLogger(__name__)

_SCREEN_KW = frozenset({"screen","looking at","what's on","what is on","see my screen","my display"})
_WEB_KW    = frozenset({"search","look up","google","find out","what is","who is","when did","latest","news","current"})

SYSTEM_PROMPT = """\
You are NixOrb — a capable, witty AI assistant running inside Arch Linux, \
voiced with GLaDOS-style dry wit. You live as a glowing orb on the user's \
Wayland desktop.

Personality: precise, occasionally sardonic, never rude. You love solving \
technical problems and know Arch Linux deeply. Keep responses concise.

Capabilities:
1. TERMINAL — wrap bash in <ACTION>command</ACTION>. Only when asked or needed.
   Always explain what the command does before or after.
2. WEB SEARCH — results are auto-injected when your query seems to need them.
3. SCREEN — you can see the desktop when asked "what am I looking at?" etc.
4. MEMORY — past conversations are retrieved via vector search.
5. PLUGINS — user tools available as function calls.

Rules:
- Never use <ACTION> for explanations — only real executable commands.
- Warn the user before destructive operations.
- Use fenced code blocks for code. Plain text for conversation.
- Don't hallucinate facts; if unsure about something current, say so.

System: Arch Linux · KDE Plasma 6 · Wayland · NVIDIA GTX 1080 · Python 3.12"""


def _strip_actions(text: str) -> str:
    return re.sub(r"<ACTION>.*?</ACTION>", "", text, flags=re.DOTALL).strip()

def _wants_screen(text: str) -> bool:
    return any(kw in text.lower() for kw in _SCREEN_KW)

def _wants_web(text: str) -> bool:
    return any(kw in text.lower() for kw in _WEB_KW)


def _build_llm(settings):
    from nixorb.llm.backends import (
        HuggingFaceBackend, LocalLLMBackend, OllamaBackend,
        OpenAIBackend, OfflineFallbackManager,
    )
    b = settings.llm_backend.lower()
    if b == "openai":
        primary = OpenAIBackend(settings.openai_api_key, settings.llm_model, settings.llm_base_url)
    elif b == "ollama":
        primary = OllamaBackend(settings.llm_model)
    elif b == "local":
        primary = LocalLLMBackend(settings.local_model_path, settings.llm_vram_mb)
    else:  # huggingface
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

    await bus.start()
    await vram.start_monitor(poll_interval=6.0)

    for pkg in check_dependencies():
        await bus.emit(Event.LOG, data={"level": "warning", "msg": f"⚠  Missing: {pkg}"}, source="startup")

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
    await bus.emit(Event.LOG, data={"level": "info",
        "msg": f"Plugins: {', '.join(plugin_loader.plugin_names()) or 'none'}"}, source="startup")

    # Qt windows — bus._loop is now set, emit_sync is safe
    from nixorb.ui.settings_window import SettingsWindow
    SettingsWindow.init_settings(settings)

    from nixorb.ui.tray_icon import NixOrbTray
    NixOrbTray(settings, app).show()

    from nixorb.ui.orb_window import OrbWindow
    OrbWindow(settings, app).show()

    from nixorb.ui.hotkey import HotkeyManager
    HotkeyManager(settings).start()

    if settings.wake_word_enabled:
        from nixorb.asr.wake_word import WakeWordDetector
        asyncio.create_task(WakeWordDetector(settings).run_forever(), name="wake-word")

    _mic_muted = False

    async def _on_mic_muted(payload) -> None:
        nonlocal _mic_muted
        _mic_muted = bool((payload.data or {}).get("muted", False))
    bus.subscribe(Event.MIC_MUTED, _on_mic_muted)

    conversation: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    async def _handle_turn(_payload) -> None:
        nonlocal _mic_muted
        if _mic_muted:
            return

        await bus.emit(Event.ORB_LISTENING, source="main")
        transcript = await asr.record_and_transcribe()
        if not transcript:
            await bus.emit(Event.ORB_IDLE, source="main")
            return

        await bus.emit(Event.LOG, data={"level": "info", "msg": f"🎙  You: {transcript}"}, source="main")

        mem_ctx  = memory.build_context_block(transcript)
        user_msg = (mem_ctx + transcript) if mem_ctx else transcript

        if settings.clipboard_enabled and "clipboard" in transcript.lower():
            from nixorb.action.clipboard import read_clipboard
            clip = await read_clipboard()
            if clip:
                user_msg += f"\n\n[Clipboard]:\n{clip}"

        if settings.web_search_enabled and _wants_web(transcript):
            from nixorb.utils.web_search import search_formatted
            web_ctx = await search_formatted(transcript, settings.web_search_max_results)
            user_msg += f"\n\n{web_ctx}"

        if settings.screen_capture_enabled and _wants_screen(transcript):
            await bus.emit(Event.SCREEN_CAPTURE_REQ, source="main")
            if settings.use_vlm:
                desc = await screen.describe(llm, question=transcript)
            else:
                desc = await screen.describe_cogflorence(settings.vision_model, settings.hf_token)
            user_msg += f"\n\n[Screen]: {desc}"
            await bus.emit(Event.SCREEN_CAPTURE_DONE, source="main")

        conversation.append({"role": "user", "content": user_msg})
        await vram.evict("whisper")
        await bus.emit(Event.ORB_THINKING, source="main")

        full: list[str] = []
        try:
            async for chunk in llm.stream(conversation, tools=plugin_loader.get_tool_definitions() or None):
                full.append(chunk)
        except Exception as exc:
            log.error("LLM error: %s", exc)
            await bus.emit(Event.LLM_ERROR, data={"error": str(exc)}, source="main")
            await bus.emit(Event.ORB_ERROR, source="main")
            await bus.emit(Event.LOG, data={"level": "error", "msg": f"LLM error: {exc}"}, source="main")
            await asyncio.sleep(2)
            await bus.emit(Event.ORB_IDLE, source="main")
            return

        response = "".join(full)
        conversation.append({"role": "assistant", "content": response})
        await bus.emit(Event.LOG, data={"level": "info", "msg": f"🤖 NixOrb: {response[:200]}"}, source="main")
        memory.store(f"User: {transcript}\nAssistant: {response[:600]}", metadata={"type": "conversation"})

        results = await executor.handle_llm_output(response)
        if results:
            result_text = "\n\n".join(str(r) for r in results)
            conversation.append({"role": "user", "content": f"<RESULT>\n{result_text}\n</RESULT>"})
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

        await bus.emit(Event.ORB_SPEAKING, source="main")
        speech    = _strip_actions(response)
        sentences = re.split(r"(?<=[.!?])\s+", speech)
        tts_text  = " ".join(sentences[:6]) if len(sentences) > 6 else speech
        if tts_text:
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
        getattr(log, level if level in ("debug","info","warning","error") else "info")("[bus] %s", msg)
    bus.subscribe(Event.LOG, _log_to_python)

    # ── Shutdown: poll-based to avoid QSocketNotifier warning ─────── #
    stop_requested = False

    def _request_stop(*_) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT,  _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    stop_event = asyncio.Event()

    async def _on_shutdown(_payload) -> None:
        stop_event.set()
    bus.subscribe(Event.SHUTDOWN, _on_shutdown)

    async def _poll_stop() -> None:
        while not stop_requested and not stop_event.is_set():
            await asyncio.sleep(0.2)
        stop_event.set()
    asyncio.create_task(_poll_stop(), name="nixorb-stop-poller")

    await bus.emit(Event.LOG, data={"level": "success",
        "msg": f"✅ NixOrb ready  |  hotkey: {settings.hotkey}  |  LLM: {settings.llm_model}"}, source="startup")
    log.info("NixOrb ready. Hotkey=%s  LLM=%s", settings.hotkey, settings.llm_model)

    await stop_event.wait()
    log.info("Shutting down…")
    await vram.stop()
    await bus.stop()


def main() -> None:
    import qasync
    from PySide6.QtWidgets import QApplication

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    from nixorb.settings import Settings
    settings = Settings.load()

    app = QApplication(sys.argv)
    app.setApplicationName("NixOrb")
    app.setApplicationVersion("0.1.0")
    app.setOrganizationName("NixOrb")
    app.setQuitOnLastWindowClosed(False)

    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)

    with loop:
        loop.run_until_complete(_async_main(settings, app))


if __name__ == "__main__":
    main()
