"""
nixorb/main.py

NixOrb daemon entry point.

Threading / event-loop model
─────────────────────────────
  Main thread  ── Qt event loop (via qasync) ── OrbWindow, SettingsWindow, Tray
  asyncio loop ── EventBus dispatch, LLM/TTS streaming coroutines
  Thread pool  ── sounddevice recording, faster-whisper inference,
                  llama-cpp generation, VRAM load/unload I/O

Start order (important for correct lock initialisation):
  1. bus.start()          ← stores the running loop reference
  2. vram.start_monitor()
  3. All AI subsystems register/init
  4. Qt windows created   ← OrbBridge.clicked() can now call emit_sync()
"""
from __future__ import annotations

import asyncio
import logging
import re
import signal
import sys
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# ── constants ────────────────────────────────────────────────────── #
SCREEN_TRIGGER_WORDS = frozenset({
    "screen", "looking at", "what's on", "what is on",
    "see this", "see my screen", "show me",
})

SYSTEM_PROMPT = """\
You are NixOrb, an intelligent AI assistant embedded in Arch Linux.
You are precise, helpful, and concise.

When asked to run a command, wrap it in <ACTION>command here</ACTION> tags.
Only use ACTION blocks when the user explicitly asks or when a task clearly
requires system interaction. Do not use them for conversation or explanation.

Respond in plain language. Use markdown for code blocks when relevant.
If you use an ACTION block, also briefly explain what it does."""


# ── helpers ──────────────────────────────────────────────────────── #
def _strip_action_blocks(text: str) -> str:
    """Remove <ACTION>…</ACTION> from text before sending to TTS."""
    return re.sub(r"<ACTION>.*?</ACTION>", "", text, flags=re.DOTALL).strip()


def _wants_screen(text: str) -> bool:
    lower = text.lower()
    return any(kw in lower for kw in SCREEN_TRIGGER_WORDS)


# ── main async coroutine ──────────────────────────────────────────── #
async def _async_main(settings, app) -> None:
    from nixorb.core.event_bus import Event, bus
    from nixorb.core.vram_manager import vram
    from nixorb.core.aur_checker import check_dependencies

    # ── 1. EventBus (must be first — stores loop reference) ─────── #
    await bus.start()
    log.info("EventBus started")

    # ── 2. VRAM monitor ─────────────────────────────────────────── #
    await vram.start_monitor(poll_interval=6.0)

    # ── 3. Dependency warnings ───────────────────────────────────── #
    missing = check_dependencies()
    for pkg in missing:
        await bus.emit(
            Event.LOG,
            data={"level": "warning", "msg": f"⚠  Missing package: {pkg}"},
            source="startup",
        )

    # ── 4. Long-term memory ──────────────────────────────────────── #
    from nixorb.memory.vector_store import VectorMemory
    memory = VectorMemory(settings.memory_dir)

    # ── 5. ASR ───────────────────────────────────────────────────── #
    from nixorb.asr.whisper_engine import WhisperEngine
    asr = WhisperEngine(settings)

    # ── 6. Vision ────────────────────────────────────────────────── #
    from nixorb.vision.screen_capture import ScreenCapture
    screen = ScreenCapture()

    # ── 7. LLM backend ───────────────────────────────────────────── #
    from nixorb.llm.backends import (
        LocalLLMBackend, OllamaBackend, OpenAIBackend, OfflineFallbackManager,
    )

    def _build_primary():
        b = settings.llm_backend.lower()
        if b == "openai":
            return OpenAIBackend(
                api_key=settings.openai_api_key,
                model=settings.llm_model,
                base_url=settings.llm_base_url,
            )
        elif b == "ollama":
            return OllamaBackend(model=settings.llm_model)
        else:
            return LocalLLMBackend(
                model_path=settings.local_model_path,
                vram_mb=settings.llm_vram_mb,
            )

    primary_llm = _build_primary()

    if settings.offline_fallback_enabled and settings.fallback_model_path:
        fallback_llm = LocalLLMBackend(
            model_path=settings.fallback_model_path, vram_mb=2_048
        )
        llm = OfflineFallbackManager(primary_llm, fallback_llm)
    else:
        llm = primary_llm  # type: ignore[assignment]

    # ── 8. TTS ───────────────────────────────────────────────────── #
    from nixorb.tts.tts_factory import build_tts
    tts = build_tts(settings)

    # ── 9. Action executor ───────────────────────────────────────── #
    from nixorb.action.executor import ActionExecutor
    executor = ActionExecutor(settings)

    # ── 10. Plugins ──────────────────────────────────────────────── #
    from nixorb.plugins.loader import PluginLoader
    plugin_loader = PluginLoader(settings.plugin_dir)
    plugin_loader.load_all()
    await bus.emit(
        Event.LOG,
        data={"level": "info",
              "msg": f"Plugins loaded: {', '.join(plugin_loader.plugin_names()) or 'none'}"},
        source="startup",
    )

    # ── 11. hypernix ─────────────────────────────────────────────── #
    from nixorb.utils.hypernix_client import HypernixClient
    _hn = HypernixClient(settings)  # noqa: unused — available for plugins

    # ── 12. Qt windows (after bus.start() so emit_sync is safe) ──── #
    from nixorb.ui.settings_window import SettingsWindow
    SettingsWindow.init_settings(settings)

    from nixorb.ui.tray_icon import NixOrbTray
    tray = NixOrbTray(settings, app)
    tray.show()

    from nixorb.ui.orb_window import OrbWindow
    orb = OrbWindow(settings, app)
    orb.show()

    # ── 13. Global hotkey ────────────────────────────────────────── #
    from nixorb.ui.hotkey import HotkeyManager
    HotkeyManager(settings).start()

    # ── 14. Wake-word (optional) ─────────────────────────────────── #
    if settings.wake_word_enabled:
        from nixorb.asr.wake_word import WakeWordDetector
        asyncio.create_task(
            WakeWordDetector(settings).run_forever(),
            name="nixorb-wake-word",
        )

    # ── 15. Clipboard ────────────────────────────────────────────── #
    from nixorb.action.clipboard import read_clipboard, write_clipboard

    # ──────────────────────────────────────────────────────────────── #
    #  Core conversation loop                                          #
    # ──────────────────────────────────────────────────────────────── #
    conversation: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    async def _handle_turn(_payload) -> None:
        """One full user-turn: listen → think → act → speak."""
        # State transitions
        await bus.emit(Event.ORB_LISTENING, source="main")

        # ── Record & transcribe ─────────────────────────────────── #
        transcript = await asr.record_and_transcribe()
        if not transcript:
            await bus.emit(Event.ORB_IDLE, source="main")
            return

        await bus.emit(
            Event.LOG,
            data={"level": "info", "msg": f"🎙  You: {transcript}"},
            source="main",
        )

        # ── Build user message ──────────────────────────────────── #
        mem_ctx  = memory.build_context_block(transcript, n=4)
        user_msg = (mem_ctx + transcript) if mem_ctx else transcript

        # ── Clipboard injection ─────────────────────────────────── #
        if settings.clipboard_enabled and "clipboard" in transcript.lower():
            clip = await read_clipboard()
            if clip:
                user_msg += f"\n\n[Clipboard content]:\n{clip}"

        # ── Screen context ──────────────────────────────────────── #
        if settings.screen_capture_enabled and _wants_screen(transcript):
            await bus.emit(Event.SCREEN_CAPTURE_REQ, source="main")
            desc = await screen.describe(primary_llm)
            user_msg += f"\n\n[Screen]: {desc}"
            await bus.emit(Event.SCREEN_CAPTURE_DONE, source="main")

        conversation.append({"role": "user", "content": user_msg})

        # ── Evict Whisper before LLM (VRAM paging) ──────────────── #
        await vram.evict("whisper")
        await bus.emit(Event.ORB_THINKING, source="main")

        # ── Stream LLM response ──────────────────────────────────── #
        full_chunks: list[str] = []
        try:
            async for chunk in llm.stream(
                conversation,
                tools=plugin_loader.get_tool_definitions() or None,
            ):
                full_chunks.append(chunk)
        except Exception as exc:
            log.error("LLM error: %s", exc)
            await bus.emit(Event.LLM_ERROR, data={"error": str(exc)}, source="main")
            await bus.emit(Event.ORB_ERROR, source="main")
            await bus.emit(
                Event.LOG,
                data={"level": "error", "msg": f"LLM error: {exc}"},
                source="main",
            )
            await asyncio.sleep(2)
            await bus.emit(Event.ORB_IDLE, source="main")
            return

        response = "".join(full_chunks)
        conversation.append({"role": "assistant", "content": response})

        await bus.emit(
            Event.LOG,
            data={"level": "info", "msg": f"🤖 NixOrb: {response[:200]}…"},
            source="main",
        )

        # ── Persist to memory ────────────────────────────────────── #
        memory.store(
            f"User: {transcript}\nAssistant: {response[:600]}",
            metadata={"type": "conversation"},
        )

        # ── Execute ACTION blocks ────────────────────────────────── #
        results = await executor.handle_llm_output(response)
        if results:
            combined = "\n\n".join(str(r) for r in results)
            conversation.append(
                {"role": "user", "content": f"<RESULT>\n{combined}\n</RESULT>"}
            )
            # Feed results back for a follow-up if there was output
            if any(r.stdout for r in results):
                followup_chunks: list[str] = []
                try:
                    async for chunk in llm.stream(conversation):
                        followup_chunks.append(chunk)
                    followup = "".join(followup_chunks)
                    if followup.strip():
                        conversation.append({"role": "assistant", "content": followup})
                        response = followup  # speak the follow-up instead
                except Exception:
                    pass  # best-effort

        # ── Copy code blocks to clipboard automatically ──────────── #
        if settings.clipboard_enabled:
            code_blocks = re.findall(r"```(?:\w+)?\n(.*?)```", response, re.DOTALL)
            if code_blocks:
                await write_clipboard(code_blocks[-1].strip())

        # ── TTS ─────────────────────────────────────────────────── #
        await bus.emit(Event.ORB_SPEAKING, source="main")
        speech = _strip_action_blocks(response)
        # Trim overly long responses for TTS (first ~3 sentences)
        sentences = re.split(r"(?<=[.!?])\s+", speech)
        speech_tts = " ".join(sentences[:6]) if len(sentences) > 6 else speech
        if speech_tts:
            await tts.speak(speech_tts)

        await bus.emit(Event.ORB_IDLE, source="main")

        # Trim conversation history (keep system + last 20 turns)
        if len(conversation) > 22:
            conversation[1:] = conversation[-20:]

    # Subscribe hotkey and wake-word both to the same handler
    bus.subscribe(Event.HOTKEY_TRIGGERED,   _handle_turn, priority=2)
    bus.subscribe(Event.WAKE_WORD_DETECTED, _handle_turn, priority=2)

    # Also subscribe LOG events to Python's logging
    async def _log_to_python(payload) -> None:
        data  = payload.data or {}
        level = data.get("level", "info")
        msg   = data.get("msg", "")
        getattr(log, level if level in ("debug", "info", "warning", "error") else "info")(
            "[bus] %s", msg
        )

    bus.subscribe(Event.LOG, _log_to_python)

    # ── Ready ─────────────────────────────────────────────────────── #
    await bus.emit(
        Event.LOG,
        data={"level": "success", "msg": "✅ NixOrb ready — press the hotkey to talk"},
        source="startup",
    )
    log.info("NixOrb daemon running. Hotkey: %s", settings.hotkey)

    # ── Graceful shutdown ─────────────────────────────────────────── #
    stop_event = asyncio.Event()

    def _on_signal(*_) -> None:
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _on_signal)
        except NotImplementedError:
            # Windows fallback (not target platform but keep clean)
            pass

    async def _on_shutdown(payload) -> None:
        stop_event.set()

    bus.subscribe(Event.SHUTDOWN, _on_shutdown)

    await stop_event.wait()

    log.info("Shutting down NixOrb…")
    await vram.stop()
    await bus.stop()
    log.info("Goodbye.")


# ── Public entry point ────────────────────────────────────────────── #
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
    app.setQuitOnLastWindowClosed(False)  # tray keeps it alive

    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)

    with loop:
        loop.run_until_complete(_async_main(settings, app))


if __name__ == "__main__":
    main()
