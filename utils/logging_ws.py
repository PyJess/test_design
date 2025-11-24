from typing import List
from fastapi import WebSocket
from datetime import datetime

active_connections: List[WebSocket] = []

async def broadcast_log(message: str):
    """Manda un log con timestamp a tutte le connessioni WebSocket attive."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    payload = f"[{timestamp}] {message}"
    for connection in active_connections:
        try:
            await connection.send_text(payload)
        except Exception:
            # Ignora connessioni chiuse o non valide
            pass
