"""
conversation_engine.py — Tier 2 Multi-Turn Conversational Context & Memory
==========================================================================
Maintains dialogue history and performs coreference resolution:
  - Tracks the "currently discussed object" (e.g. Person #1)
  - Resolves pronouns and follow-up fragments:
      "What color is the shirt?" -> "Purple."
      "And the pants?"           -> resolves to lower garments of Person #1
      "Is he moving?"            -> resolves to motion state of Person #1
      "Does he have earrings?"   -> resolves to ear accessories of Person #1
100% local, pure Python, zero cloud APIs.
"""

import time
import re


class ConversationEngine:
    """Manages multi-turn dialogue memory and conversational coreference."""

    def __init__(self, max_history: int = 15):
        self.history: list[dict] = []
        self.max_history = max_history
        self.last_target_id: int | None = None
        self.last_target_class: str = "person"

    def record_turn(self, query: str, answer: str, target_id: int | None = None, target_cls: str = "person"):
        """Record an interaction turn into conversation history."""
        self.history.append({
            "timestamp": time.time(),
            "query":     query,
            "answer":    answer,
            "target_id": target_id,
            "target_cls": target_cls,
        })
        if len(self.history) > self.max_history:
            self.history.pop(0)

        if target_id is not None:
            self.last_target_id = target_id
            self.last_target_class = target_cls

    def resolve_target(self, query: str, scene: dict) -> dict | None:
        """
        Determine which object in the current scene the user is asking about.
        Resolves explicit IDs, pronouns ('he', 'she', 'they', 'it'), or defaults
        to the most recently discussed object.
        """
        objects = scene.get("objects", [])
        if not objects:
            return None

        q = query.lower().strip()

        # 1. Explicit ID in query: e.g. "person 2", "#1", "id 3"
        id_match = re.search(r"(?:id|#|number|no\.?)\s*(\d+)", q)
        if id_match:
            req_id = int(id_match.group(1))
            for obj in objects:
                if obj["id"] == req_id:
                    return obj

        # 2. Pronouns or elliptical follow-ups ("he", "she", "they", "that person", "and the pants?")
        is_follow_up = any(w in q for w in [
            "he", "she", "they", "him", "her", "his", "their", "that", "this",
            "and ", "what about", "how about", "also", "too"
        ])

        if is_follow_up and self.last_target_id is not None:
            # Check if last discussed object is still in current scene
            for obj in objects:
                if obj["id"] == self.last_target_id:
                    return obj

        # 3. If class mentioned (e.g. "car", "dog"), match by class
        for obj in objects:
            cls = obj["class"].lower()
            if cls in q:
                return obj

        # 4. Default: return first person if available, else first object
        for obj in objects:
            if obj["class"] == "person":
                return obj

        return objects[0]

    def get_context_summary(self) -> str:
        """Return a formatted string of recent conversation context."""
        if not self.history:
            return "No previous conversational turns."
        lines = []
        for i, turn in enumerate(self.history[-3:], 1):
            lines.append(f"Q: {turn['query']} -> A: {turn['answer']}")
        return " | ".join(lines)
