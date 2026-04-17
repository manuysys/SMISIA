"""
AGRILION — Chatbot Demo
=========================
Standalone test script — runs without the full AGRILION pipeline.

Usage:
    # With API key (real LLM):
    OPENROUTER_API_KEY=sk-or-v1-... python demo_chatbot.py

    # Without API key (fallback mode):
    python demo_chatbot.py
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env variables before anything else
load_dotenv(Path(__file__).parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.chatbot import ChatbotService, build_system_context


def main():
    print("=" * 60)
    print("  AGRILION — Chatbot Demo")
    print("=" * 60)

    # Simulate a silo in WARNING state
    mock_context = build_system_context(
        silo_id="SILO_001",
        sensor_values={
            "temperature": 31.4,
            "humidity": 76.2,
            "co2": 680.0,
        },
        risk_score=62,
        risk_level="WARNING",
        alerts=[
            {
                "level": "WARNING",
                "category": "hongos",
                "message": "Riesgo de hongos: temperatura 31.4°C y humedad 76.2%",
                "recommendation": "Verificar sellado del silobolsa. Monitorear evolución en 12h.",
            }
        ],
        predictions={
            "temperature": 32.1,
            "humidity": 77.5,
            "co2": 710.0,
        },
    )

    svc = ChatbotService()
    session = "demo-session"

    questions = [
        "Is my silo in danger right now?",
        "Why is the current humidity dangerous?",
        "What does the WARNING alert mean and what should I do?",
        "What will happen with the temperature in the next hours?",
    ]

    for q in questions:
        print(f"\n[User]: {q}")
        resp = svc.chat(q, silo_id="SILO_001", session_id=session, context_override=mock_context)
        clean_response = resp.response.encode('ascii', 'replace').decode('ascii')
        print(f"[AGRILION]: {clean_response}")
        print(f"   ({resp.latency_ms:.0f}ms)")
        if resp.error:
            print(f"   [!] Error: {resp.error}")

    print("\n" + "=" * 60)
    print("Demo complete.")
    if not os.getenv("OPENROUTER_API_KEY") and not os.getenv("HUGGINGFACE_API_KEY"):
        print("\n[INFO] To enable real AI responses:")
        print("   1. Get a free key at https://openrouter.ai")
        print("   2. Set: OPENROUTER_API_KEY=sk-or-v1-... python demo_chatbot.py")


if __name__ == "__main__":
    main()
