"""
nixorb/main.py

Daemon bootstrap. Bridges Qt's event loop with asyncio using
qasync (PySide6-compatible). All subsystems start here.

Threading model:
  Main thread  → Qt GUI (orb window, settings, tray)
  asyncio loop → EventBus dispatch, LLM streaming, TTS streaming
  ThreadPool   → Whisper transcription, VRAM loads, recording I/O
"""
from __future__ import annotations

import asyncio
import logging
import signal
import sys
from pathlib import Path

import qasync  # pip install qasync
from PySide6.QtWidgets import QApplication

from nixorb.core.event_bus import Event, bus
from nixorb.core.vram_manager import vram
from nixorb.core.aur_checker import check_dependencies
from nixorb.settings import Settings

log = logging.getLogger(__name__)


async def _async_main(settings: Settings, app: QApplication) -> None:
    """
    Master async coroutine. Starts all subsystems in dependency order.
    """
    # 1. EventBus
    await bus.start()
    log.info("EventBus started")

    # 2. VRAM monitor
    await vram.start_monitor(poll_interval=5.0)

    # 3. Long-term memory
    from nixorb.memory.vector_store import VectorMemory
    memory = VectorMemory()

    # 4. ASR engine
    from nixorb.asr.whisper_engine import WhisperEngine
    asr = WhisperEngine(settings)

    # 5. LLM backend (factory based on settings)
    from nixorb.llm.backends import (
        OpenAIBackend, LocalLLMBackend, OllamaBackend, OfflineFallbackManager
    )
    if settings.llm_backend == "openai":
        primary_llm = OpenAIBackend(
            api_key=settings.openai_api_key,
            model=settings.llm_model,
            base_url=settings.llm_base_url,
        )
    elif settings.llm_backend == "ollama":
        primary_llm = OllamaBackend(model=settings.llm_model)
    else:
        primary_llm = LocalLLMBackend(model_path=settings.local_model_path)

    fallback_llm = LocalLLMBackend(
        model_path=settings.fallback_model_path or settings.local_model_path,
        vram_mb=2048,
    )
    llm = OfflineFallbackManager(primary_llm, fallback_llm)

    # 6. TTS engine
    from nixorb.tts.tts_factory import build_tts
    tts = build_tts(settings)

    # 7. Action executor
    from nixorb.action.executor import ActionExecutor
    executor = ActionExecutor(settings)

    # 8. Plugins
    from nixorb.plugins.loader import plugin_loader
    plugin_loader.load_all()

    # 9. Wake-word (optional)
    if settings.wake_word_enabled:
        from nixorb.asr.wake_word import WakeWordDetector
        wake = WakeWordDetector(settings)
        asyncio.create_task(wake.run_forever(), name="wake-word")

    # 10. System tray
    from nixorb.ui.tray_icon import NixOrbTray
    tray = NixOrbTray(settings, app)
    tray.show()

    # 11. Orb window
    from nixorb.ui.orb_window import OrbWindow
    orb = OrbWindow(settings, app)
    orb.show()

    # 12. Hotkey
    from nixorb.ui.hotkey import HotkeyManager
    hotkey_mgr = HotkeyManager(settings)
    hotkey_mgr.start()

    # 13. hypernix client
    from nixorb.utils.hypernix_client import HypernixClient
    hn_client = HypernixClient(settings)

    # ---------------------------------------------------------------- #
    #  Core conversation loop                                           #
    # ---------------------------------------------------------------- #
    SYSTEM_PROMPT = """You are NixOrb, a helpful AI assistant embedded in Arch Linux.
You have access to the user's terminal via <ACTION>command</ACTION> blocks.
Only use ACTION blocks when explicitly asked or when a task requires system interaction.
Always be concise. Respond in plain language unless showing code."""

    conversation: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    async def handle_hotkey(payload) -> None:
        await bus.emit(Event.ORB_LISTENING, source="main")
        transcript = await asr.record_and_transcribe()

        if not transcript:
            await bus.emit(Event.ORB_IDLE, source="main")
            return

        log.info("User: %s", transcript)
        await bus.emit(
            Event.LOG,
            data={"level": "info", "msg": f"🎙 You: {transcript}"},
            source="main",
        )

        # Memory context injection
        mem_ctx = memory.build_context_block(transcript)
        user_content = mem_ctx + transcript if mem_ctx else transcript

        # Screen context (if requested)
        if any(kw in transcript.lower() for kw in
               ["what am i looking at", "what's on screen", "screen"]):
            from nixorb.vision.screen_capture import ScreenCapture
            sc = ScreenCapture()
            screen_desc = await sc.describe(primary_llm)
            user_content += f"\n[Screen content: {screen_desc}]"

        conversation.append({"role": "user", "content": user_content})

        # Evict whisper before LLM runs
        await vram.evict("whisper")
        await bus.emit(Event.ORB_THINKING, source="main")

        # Stream LLM response
        full_response = []
        try:
            async for chunk in llm.stream(
                conversation,
                tools=plugin_loader.get_tool_definitions() or None,
            ):
                full_response.append(chunk)
        except Exception as exc:
            log.error("LLM error: %s", exc)
            await bus.emit(Event.ORB_ERROR, source="main")
            await bus.emit(
                Event.LOG,
                data={"level": "error", "msg": f"LLM error: {exc}"},
                source="main",
            )
            return

        response_text = "".join(full_response)
        conversation.append({"role": "assistant", "content": response_text})

        # Store in long-term memory
        memory.store(f"User: {transcript}\nAssistant: {response_text[:500]}")

        # Execute any ACTION blocks
        results = await executor.handle_llm_output(response_text)
        if results:
            result_text = "\n".join(str(r) for r in results)
            conversation.append({
                "role": "user",
                "content": f"<RESULT>\n{result_text}\n</RESULT>"
            })

        # TTS
        await bus.emit(Event.ORB_SPEAKING, source="main")
        # Strip ACTION blocks before speaking
        import re
        speech_text = re.sub(r"<ACTION>.*?</ACTION>", "", response_text,
                              flags=re.DOTALL).strip()
        if speech_text:
            await tts.speak(speech_text)

        await bus.emit(Event.ORB_IDLE, source="main")

    bus.subscribe(Event.HOTKEY_TRIGGERED, handle_hotkey)
    bus.subscribe(Event.WAKE_WORD_DETECTED, handle_hotkey)

    # Graceful shutdown
    stop_event = asyncio.Event()

    def _sighandler(*_):
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _sighandler)

    log.info("NixOrb daemon running. Press Ctrl+C to exit.")
    await bus.emit(
        Event.LOG,
        data={"level": "info", "msg": "✅ NixOrb started"},
        source="main",
    )

    await stop_event.wait()

    # Cleanup
    log.info("Shutting down...")
    await bus.emit(Event.SHUTDOWN, source="main")
    await vram.stop()
    await bus.stop()


def main() -> None:
    # AUR dependency check (non-blocking warnings)
    missing = check_dependencies()
    if missing:
        print(f"[WARNING] Missing packages: {', '.join(missing)}")
        print("Install them for full functionality.")

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    settings = Settings.load()

    app = QApplication(sys.argv)
    app.setApplicationName("NixOrb")
    app.setQuitOnLastWindowClosed(False)  # tray keeps it alive

    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)

    with loop:
        loop.run_until_complete(_async_main(settings, app))


if __name__ == "__main__":
    main()
