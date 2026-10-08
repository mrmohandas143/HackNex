"""
qa_engine.py  —  Tier 1 Rule-based Q&A
Answers questions purely from YOLO detections + color analysis.
No LLM, no API, no internet required.
"""
import re
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Synonym groups: any alias maps to the canonical COCO class name
# ─────────────────────────────────────────────────────────────────────────────
SYNONYMS: dict[str, str] = {
    # people / humans
    "people":     "person",
    "persons":    "person",
    "human":      "person",
    "humans":     "person",
    "man":        "person",
    "men":        "person",
    "woman":      "person",
    "women":      "person",
    "guy":        "person",
    "guys":       "person",
    "girl":       "person",
    "girls":      "person",
    "boy":        "person",
    "boys":       "person",
    "child":      "person",
    "children":   "person",
    "kid":        "person",
    "kids":       "person",
    # clothing → person (we detect the person and report their clothing color)
    "shirt":      "person",
    "t-shirt":    "person",
    "tshirt":     "person",
    "top":        "person",
    "jacket":     "person",
    "hoodie":     "person",
    "clothes":    "person",
    "clothing":   "person",
    "trousers":   "person",
    "pants":      "person",
    "jeans":      "person",
    # common aliases
    "automobile": "car",
    "automobiles":"car",
    "vehicle":    "car",
    "vehicles":   "car",
    "motorbike":  "motorcycle",
    "bike":       "bicycle",
    "bikes":      "bicycle",
    "sofa":       "couch",
    "tv":         "tv",
    "television": "tv",
    "phone":      "cell phone",
    "phones":     "cell phone",
    "mobile":     "cell phone",
    "cellphone":  "cell phone",
    "laptop":     "laptop",
    "laptops":    "laptop",
    "bag":        "handbag",
    "bags":       "handbag",
    "plant":      "potted plant",
    "plants":     "potted plant",
    "table":      "dining table",
    "bottle":     "bottle",
    "bottles":    "bottle",
    "cup":        "cup",
    "cups":       "cup",
}

# Query keywords that indicate clothing-color intent (special handling)
CLOTHING_WORDS = {"shirt", "t-shirt", "tshirt", "top", "jacket",
                  "hoodie", "clothes", "clothing", "trousers",
                  "pants", "jeans", "wearing", "dress"}


class QAEngine:
    """
    Tier-1 Q&A engine: interprets natural-language questions and
    produces answers from structured scene context built by SceneBuilder.
    """

    # ── Public entry-point ────────────────────────────────────────────────────

    def answer(self, query: str, scene: dict, frame=None) -> str:
        q = query.lower().strip().rstrip("?").strip()

        # ── Route to the right handler ────────────────────────────────────
        if self._match(q, ["what is in", "what's in", "what do you see",
                           "what can you see", "describe", "what is happening",
                           "what's going on", "tell me what", "what objects",
                           "what is there", "show me", "list"]):
            return self._describe(scene)

        if self._match(q, ["how many objects", "how many things", "total objects",
                           "total count", "how many items"]):
            return f"There are {scene['total_count']} object(s) detected in the frame."

        count_cls = self._extract(q, [
            r"how many (\w[\w\s-]*?)(?:\s+are|\s+is|\s+do|\s+in|$)",
            r"count (?:the |all )?(\w[\w\s-]*)",
            r"number of (\w[\w\s-]*)",
        ])
        if count_cls:
            return self._count(count_cls.strip(), scene)

        color_obj = self._extract(q, [
            r"(?:what|which) (?:color|colour) (?:is|are) (?:the )?(.+)",
            r"(?:color|colour) of (?:the )?(.+)",
            r"what (?:is|are) (?:the )?(.+?) (?:color|colour)",
            r"which (?:is|are) (?:the )?(.+?) (?:color|colour)",
            r"tell me the (?:color|colour) of (?:the )?(.+)",
        ])
        if color_obj:
            return self._color(color_obj.strip(), scene, q)

        where_obj = self._extract(q, [
            r"where is (?:the )?(.+)",
            r"where are (?:the )?(.+)",
            r"position of (?:the )?(.+)",
            r"locate (?:the )?(.+)",
        ])
        if where_obj:
            return self._where(where_obj.strip(), scene)

        exist_obj = self._extract(q, [
            r"is there (?:a |an )?(.+)",
            r"are there (?:any )?(.+)",
            r"(?:do you|can you) see (?:a |an |any )?(.+)",
            r"(?:is|are) (?:there )?(?:a |an |any )?(.+?) (?:visible|present|there)",
        ])
        if exist_obj:
            return self._exists(exist_obj.strip(), scene)

        # Fallback
        return self._fallback(q, scene)

    # ── Handlers ─────────────────────────────────────────────────────────────

    def _describe(self, scene: dict) -> str:
        if not scene["objects"]:
            return "No objects are currently detected in the frame."

        lines = ["Here is what I see in the frame:\n"]
        for obj in scene["objects"]:
            color_part = (
                f"{obj['color']} " if obj["color"] not in ("unknown", "colorful") else ""
            )
            lines.append(
                f"  - {color_part}{obj['class']}  "
                f"[Track ID #{obj['id']}]  --  {obj['position']} of frame"
            )

        counts = scene["object_counts"]
        summary = ", ".join(f"{v} {k}(s)" for k, v in counts.items())
        lines.append(f"\nSummary: {summary}")
        return "\n".join(lines)

    def _count(self, raw_cls: str, scene: dict) -> str:
        cls = self._resolve(raw_cls)

        # Direct match
        cnt = scene["object_counts"].get(cls, 0)
        if cnt:
            return f"There are {cnt} {cls}(s) in the frame."

        # Partial / fuzzy match
        for k, v in scene["object_counts"].items():
            if cls in k or k in cls:
                return f"There are {v} {k}(s) in the frame."

        return (
            f"I don't see any '{raw_cls}' in the frame.  "
            + self._visible_hint(scene)
        )

    def _color(self, raw_obj: str, scene: dict, original_q: str) -> str:
        """Handle 'what color is the X?' -- with clothing special case."""
        is_clothing = any(w in raw_obj for w in CLOTHING_WORDS) or \
                      any(w in original_q for w in CLOTHING_WORDS)

        cls = self._resolve(raw_obj)
        matches = self._find(cls, scene)

        if not matches:
            return (
                f"I don't see any '{raw_obj}' in the frame.  "
                + self._visible_hint(scene)
            )

        if is_clothing and cls == "person":
            # Report person's torso/clothing color
            clothing_word = "t-shirt" if any(w in original_q or w in raw_obj for w in ["tshirt", "t-shirt", "shirt"]) else "clothing"
            if len(matches) == 1:
                obj = matches[0]
                c = obj.get("clothing_color", obj["color"])
                return (
                    f"The person (ID #{obj['id']})'s {clothing_word} appears to be "
                    f"{c}."
                )
            parts = ", ".join(
                f"ID #{o['id']}: {o.get('clothing_color', o['color'])}" for o in matches
            )
            return f"Multiple people detected -- {clothing_word} colors: {parts}."

        if len(matches) == 1:
            obj = matches[0]
            return (
                f"The {obj['class']} (ID #{obj['id']}) is predominantly "
                f"{obj['color']} colored."
            )

        parts = ", ".join(f"ID #{o['id']}: {o['color']}" for o in matches)
        return f"{len(matches)} {cls}(s) detected -- colors: {parts}."

    def _where(self, raw_obj: str, scene: dict) -> str:
        cls     = self._resolve(raw_obj)
        matches = self._find(cls, scene)

        if not matches:
            return (
                f"I don't see any '{raw_obj}' in the frame.  "
                + self._visible_hint(scene)
            )
        if len(matches) == 1:
            obj = matches[0]
            return (
                f"The {obj['class']} (ID #{obj['id']}) is at the "
                f"{obj['position']} of the frame."
            )

        parts = ", ".join(f"ID #{o['id']} -> {o['position']}" for o in matches)
        return f"Found {len(matches)} {cls}(s): {parts}."

    def _exists(self, raw_obj: str, scene: dict) -> str:
        cls     = self._resolve(raw_obj)
        matches = self._find(cls, scene)

        if matches:
            n    = len(matches)
            verb = "is" if n == 1 else "are"
            return f"Yes, there {verb} {n} {cls}(s) visible in the frame."

        return (
            f"No, I don't see any '{raw_obj}' right now.  "
            + self._visible_hint(scene)
        )

    def _fallback(self, q: str, scene: dict) -> str:
        if not scene["objects"]:
            return (
                "No objects are detected right now.  "
                "Try pointing the camera at something and press [Q] again."
            )
        return (
            f"I'm not sure about that query -- but here's what I see:\n"
            f"{scene['description']}\n\n"
            "Try asking:\n"
            "  - \"What is in the frame?\"\n"
            "  - \"How many people are there?\"\n"
            "  - \"What color is the car?\"\n"
            "  - \"Is there a dog?\"\n"
            "  - \"Where is the person?\""
        )

    # ── Utility helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _match(text: str, keywords: list[str]) -> bool:
        return any(kw in text for kw in keywords)

    @staticmethod
    def _extract(text: str, patterns: list[str]) -> str | None:
        for pat in patterns:
            m = re.search(pat, text)
            if m:
                return m.group(1).strip()
        return None

    @staticmethod
    def _resolve(raw: str) -> str:
        """Map synonyms/aliases to canonical COCO class names."""
        low = raw.lower().strip()
        if low in SYNONYMS:
            return SYNONYMS[low]
        key = low.rstrip("s")
        if key in SYNONYMS:
            return SYNONYMS[key]
        return key if key else low

    @staticmethod
    def _find(cls: str, scene: dict) -> list[dict]:
        """Return scene objects whose class fuzzy-matches *cls*."""
        out = []
        for obj in scene["objects"]:
            oc = obj["class"].lower()
            if cls == oc or cls in oc or oc in cls:
                out.append(obj)
        return out

    @staticmethod
    def _visible_hint(scene: dict) -> str:
        if scene["object_counts"]:
            visible = ", ".join(scene["object_counts"].keys())
            return f"Currently visible: {visible}."
        return "The frame appears to be empty."
