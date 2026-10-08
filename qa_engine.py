"""
qa_engine.py — Tier 2 Visual Reasoning & Conversational Q&A Engine
==================================================================
Supports:
  - Open-ended visual attribute queries (earrings, glasses, hats, weapons, carried objects)
  - Full outfit inspection (shirt color, pants color, patterns)
  - Posture & Action understanding (sitting vs standing, moving vs still)
  - Temporal & Motion dynamics (dwell time, motion direction, entry/exit events)
  - Multi-turn conversational dialogue memory & pronoun coreference resolution
100% open-source, local, offline. Zero servers, zero API keys.
"""

import re
from conversation_engine import ConversationEngine


# ─────────────────────────────────────────────────────────────────────────────
# Synonym groups
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
    # clothing aliases
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
    # common objects
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

CLOTHING_WORDS = {
    "shirt", "t-shirt", "tshirt", "top", "jacket", "hoodie",
    "clothes", "clothing", "trousers", "pants", "jeans", "wearing", "dress"
}


class QAEngine:
    """
    Tier 2 Intelligent Vision-Language Q&A Engine.
    Combines rule-based classification, deep anatomical attribute inspection,
    temporal dynamics, and multi-turn conversational memory.
    """

    def __init__(self, temporal_tracker=None):
        self.conv_engine = ConversationEngine()
        self.temporal_tracker = temporal_tracker

    def set_temporal_tracker(self, tracker):
        self.temporal_tracker = tracker

    # ── Main Entry-Point ──────────────────────────────────────────────────────

    def answer(self, query: str, scene: dict, frame=None) -> str:
        q = query.lower().strip().rstrip("?").strip()

        # ── 0. Check for conversational recall ────────────────────────────
        if self._match(q, ["what did i ask", "repeat", "previous question", "what was that"]):
            ans = self._handle_history_recall()
            self.conv_engine.record_turn(query, ans)
            return ans

        # Resolve target object for conversational coreference
        target_obj = self.conv_engine.resolve_target(query, scene)
        tid = target_obj["id"] if target_obj else None

        # ── 0b. Conversational fragment follow-ups ────────────────────────
        frag_match = self._extract(q, [
            r"(?:and |how about |what about )(?:the )?(\w[\w\s-]*)",
        ])
        if frag_match and target_obj:
            item = frag_match.strip()
            if any(w in item for w in ["shirt", "tshirt", "t-shirt", "top"]):
                c = target_obj.get("clothing_color", "unknown")
                ans = f"The person (ID #{target_obj['id']})'s shirt appears to be {c}."
                self.conv_engine.record_turn(query, ans, target_id=tid)
                return ans
            elif any(w in item for w in ["pant", "pants", "trouser", "trousers", "jeans"]):
                ans = self._handle_pants(scene, target_obj)
                self.conv_engine.record_turn(query, ans, target_id=tid)
                return ans
            elif any(w in item for w in ["hat", "cap", "headwear"]):
                ans = self._handle_hat(scene, target_obj)
                self.conv_engine.record_turn(query, ans, target_id=tid)
                return ans
            elif any(w in item for w in ["earring", "earrings", "earing", "earings"]):
                ans = self._handle_earrings(scene, target_obj)
                self.conv_engine.record_turn(query, ans, target_id=tid)
                return ans
            elif any(w in item for w in ["glass", "glasses", "spectacles"]):
                ans = self._handle_glasses(scene, target_obj)
                self.conv_engine.record_turn(query, ans, target_id=tid)
                return ans

        # ── 1. Accessories: Earrings ──────────────────────────────────────
        if self._match(q, ["earring", "earrings", "earing", "earings", "ear ring", "ear rings", "on the ear", "in the ear"]):
            ans = self._handle_earrings(scene, target_obj)
            self.conv_engine.record_turn(query, ans, target_id=tid)
            return ans

        # ── 2. Accessories: Glasses / Eyewear ─────────────────────────────
        if self._match(q, ["glass", "glasses", "spectacle", "spectacles", "eyewear", "shades", "sunglasses"]):
            ans = self._handle_glasses(scene, target_obj)
            self.conv_engine.record_turn(query, ans, target_id=tid)
            return ans

        # ── 3. Headwear / Hat ─────────────────────────────────────────────
        if self._match(q, ["hat", "cap", "headwear", "beanie", "helmet", "on the head"]):
            ans = self._handle_hat(scene, target_obj)
            self.conv_engine.record_turn(query, ans, target_id=tid)
            return ans

        # ── 4. Weapons & Carried Objects ──────────────────────────────────
        if self._match(q, ["weapon", "weapons", "gun", "knife", "blade", "armed",
                           "holding", "carrying", "in hand", "in the hand"]):
            ans = self._handle_weapons_and_items(scene, target_obj, q)
            self.conv_engine.record_turn(query, ans, target_id=tid)
            return ans

        # ── 5. Posture & Action: Standing, Sitting, Moving ────────────────
        if self._match(q, ["sitting", "standing", "crouching", "posture", "doing",
                           "is he sitting", "is she sitting", "is the person sitting"]):
            ans = self._handle_posture(scene, target_obj)
            self.conv_engine.record_turn(query, ans, target_id=tid)
            return ans

        # ── 6. Temporal & Motion Dynamics ─────────────────────────────────
        if self._match(q, ["how long", "dwell time", "how much time", "since when"]):
            ans = self._handle_dwell_time(scene, target_obj)
            self.conv_engine.record_turn(query, ans, target_id=tid)
            return ans

        if self._match(q, ["moving", "direction", "which direction", "is he moving",
                           "are they moving", "movement", "speed", "walking"]):
            ans = self._handle_motion(scene, target_obj)
            self.conv_engine.record_turn(query, ans, target_id=tid)
            return ans

        if self._match(q, ["did anyone enter", "did anyone leave", "who entered", "who left",
                           "recent events", "what happened"]):
            ans = self._handle_events()
            self.conv_engine.record_turn(query, ans)
            return ans

        # ── 7. Pants / Lower Clothing ─────────────────────────────────────
        if self._match(q, ["pants", "pant", "trousers", "trouser", "jeans", "shorts", "lower clothing", "bottom"]):
            ans = self._handle_pants(scene, target_obj)
            self.conv_engine.record_turn(query, ans, target_id=tid)
            return ans

        # ── 8. What is the person wearing? (Full Outfit) ──────────────────
        if self._match(q, ["what is the person wearing", "what are they wearing", "what is he wearing",
                           "what is she wearing", "describe outfit", "outfit", "wearing"]):
            ans = self._handle_full_outfit(scene, target_obj)
            self.conv_engine.record_turn(query, ans, target_id=tid)
            return ans

        # ── 9. Scene Description & Overview ───────────────────────────────
        if self._match(q, ["what is in", "what's in", "what do you see", "what can you see",
                           "describe", "what is happening", "what's going on", "tell me what",
                           "what objects", "what is there", "show me", "list"]):
            ans = self._describe(scene)
            self.conv_engine.record_turn(query, ans)
            return ans

        # ── 10. Counting Queries ──────────────────────────────────────────
        if self._match(q, ["how many objects", "how many things", "total objects",
                           "total count", "how many items"]):
            ans = f"There are {scene.get('total_count', 0)} object(s) detected in the frame."
            self.conv_engine.record_turn(query, ans)
            return ans

        count_cls = self._extract(q, [
            r"how many (\w[\w\s-]*?)(?:\s+are|\s+is|\s+do|\s+in|$)",
            r"count (?:the |all )?(\w[\w\s-]*)",
            r"number of (\w[\w\s-]*)",
        ])
        if count_cls:
            ans = self._count(count_cls.strip(), scene)
            self.conv_engine.record_turn(query, ans)
            return ans

        # ── 11. Color Queries ─────────────────────────────────────────────
        color_obj = self._extract(q, [
            r"(?:what|which) (?:color|colour) (?:is|are) (?:the )?(.+)",
            r"(?:color|colour) of (?:the )?(.+)",
            r"what (?:is|are) (?:the )?(.+?) (?:color|colour)",
            r"which (?:is|are) (?:the )?(.+?) (?:color|colour)",
            r"tell me the (?:color|colour) of (?:the )?(.+)",
        ])
        if color_obj:
            ans = self._color(color_obj.strip(), scene, q, target_obj)
            self.conv_engine.record_turn(query, ans, target_id=tid)
            return ans

        # ── 12. Location / Position Queries ───────────────────────────────
        where_obj = self._extract(q, [
            r"where is (?:the )?(.+)",
            r"where are (?:the )?(.+)",
            r"position of (?:the )?(.+)",
            r"locate (?:the )?(.+)",
        ])
        if where_obj:
            ans = self._where(where_obj.strip(), scene)
            self.conv_engine.record_turn(query, ans, target_id=tid)
            return ans

        # ── 13. Existence Verification ────────────────────────────────────
        exist_obj = self._extract(q, [
            r"is there (?:a |an )?(.+)",
            r"are there (?:any )?(.+)",
            r"(?:do you|can you) see (?:a |an |any )?(.+)",
            r"(?:is|are) (?:there )?(?:a |an |any )?(.+?) (?:visible|present|there)",
        ])
        if exist_obj:
            ans = self._exists(exist_obj.strip(), scene)
            self.conv_engine.record_turn(query, ans)
            return ans

        # ── 14. Fallback ──────────────────────────────────────────────────
        ans = self._fallback(q, scene)
        self.conv_engine.record_turn(query, ans)
        return ans

    # ── Tier 2 Specialized Handlers ──────────────────────────────────────────

    def _handle_earrings(self, scene: dict, target_obj: dict | None) -> str:
        persons = [o for o in scene.get("objects", []) if o["class"] == "person"]
        if not persons:
            return "No persons are visible in the frame to inspect for earrings."

        obj = target_obj if target_obj and target_obj["class"] == "person" else persons[0]
        attrs = obj.get("attributes")
        if not attrs:
            return f"Person (ID #{obj['id']}) is detected, but resolution was insufficient for ear inspection."

        ear_info = attrs.get("earrings", {})
        if ear_info.get("detected"):
            return f"Yes, earrings/ear jewelry detected on Person (ID #{obj['id']}): {ear_info.get('detail', 'Reflective jewelry accents visible')}."
        return f"No visible earrings or ear accessories detected on Person (ID #{obj['id']})."

    def _handle_glasses(self, scene: dict, target_obj: dict | None) -> str:
        persons = [o for o in scene.get("objects", []) if o["class"] == "person"]
        if not persons:
            return "No persons are visible in the frame to inspect for eyeglasses."

        obj = target_obj if target_obj and target_obj["class"] == "person" else persons[0]
        attrs = obj.get("attributes")
        if attrs and attrs.get("glasses", {}).get("detected"):
            return f"Yes, Person (ID #{obj['id']}) appears to be wearing eyeglasses/spectacles."
        return f"Person (ID #{obj['id']}) does not appear to be wearing glasses."

    def _handle_hat(self, scene: dict, target_obj: dict | None) -> str:
        persons = [o for o in scene.get("objects", []) if o["class"] == "person"]
        if not persons:
            return "No persons are visible in the frame to inspect for headwear."

        obj = target_obj if target_obj and target_obj["class"] == "person" else persons[0]
        attrs = obj.get("attributes")
        if attrs and attrs.get("hat", {}).get("detected"):
            col = attrs["hat"]["color"]
            return f"Yes, Person (ID #{obj['id']}) is wearing headwear ({col} colored cap/hat)."
        return f"Person (ID #{obj['id']}) is not wearing a hat or headwear."

    def _handle_weapons_and_items(self, scene: dict, target_obj: dict | None, query: str) -> str:
        persons = [o for o in scene.get("objects", []) if o["class"] == "person"]
        if not persons:
            return "No persons are detected in the frame."

        obj = target_obj if target_obj and target_obj["class"] == "person" else persons[0]
        attrs = obj.get("attributes")
        if not attrs:
            return f"No weapons or suspicious items detected on Person (ID #{obj['id']})."

        w_info = attrs.get("weapon", {})
        if w_info.get("detected"):
            return f"ALERT: {w_info.get('details', 'Suspicious elongated object/potential weapon detected near hand area!')}"

        h_info = attrs.get("holding_item", {})
        if "hold" in query or "carry" in query:
            if h_info.get("detected"):
                return f"Person (ID #{obj['id']}): {h_info.get('details', 'Holding a small item')}."
            return f"Person (ID #{obj['id']}) does not appear to be holding anything. Hands are empty."

        return f"No weapons detected on Person (ID #{obj['id']}). The hands and body perimeter appear clear."

    def _handle_posture(self, scene: dict, target_obj: dict | None) -> str:
        persons = [o for o in scene.get("objects", []) if o["class"] == "person"]
        if not persons:
            return "No persons are visible in the frame."

        obj = target_obj if target_obj and target_obj["class"] == "person" else persons[0]
        posture = obj.get("posture", "standing")
        motion = obj.get("motion_direction", "stationary")
        return f"Person (ID #{obj['id']}) is currently {posture} ({motion})."

    def _handle_dwell_time(self, scene: dict, target_obj: dict | None) -> str:
        if not scene.get("objects"):
            return "No objects are currently in the frame."

        obj = target_obj if target_obj else scene["objects"][0]
        dwell = obj.get("dwell_time", 0.0)
        return f"{obj['class'].capitalize()} (ID #{obj['id']}) has been in the frame for {dwell:.1f} seconds."

    def _handle_motion(self, scene: dict, target_obj: dict | None) -> str:
        if not scene.get("objects"):
            return "No objects are currently detected."

        obj = target_obj if target_obj else scene["objects"][0]
        direction = obj.get("motion_direction", "stationary")
        speed = obj.get("velocity_px_s", 0.0)

        if obj.get("is_moving", False):
            return f"{obj['class'].capitalize()} (ID #{obj['id']}) is actively moving: {direction} (speed: {speed:.0f} px/s)."
        return f"{obj['class'].capitalize()} (ID #{obj['id']}) is currently stationary / standing still."

    def _handle_events(self) -> str:
        if not self.temporal_tracker:
            return "Temporal event tracker is not active."
        events = self.temporal_tracker.get_recent_events(seconds=45.0)
        if not events:
            return "No entries or departures recorded in the last 45 seconds."
        summary = " | ".join(e["message"] for e in events[-4:])
        return f"Recent events: {summary}"

    def _handle_pants(self, scene: dict, target_obj: dict | None) -> str:
        persons = [o for o in scene.get("objects", []) if o["class"] == "person"]
        if not persons:
            return "No persons are visible in the frame."

        obj = target_obj if target_obj and target_obj["class"] == "person" else persons[0]
        pants_col = obj.get("pants_color", "unknown")
        if pants_col and pants_col not in ("unknown", "colorful"):
            return f"The person (ID #{obj['id']})'s pants/trousers appear to be {pants_col}."
        return f"Person (ID #{obj['id']})'s lower garment color is indistinct or not clearly visible."

    def _handle_full_outfit(self, scene: dict, target_obj: dict | None) -> str:
        persons = [o for o in scene.get("objects", []) if o["class"] == "person"]
        if not persons:
            return "No persons are visible in the frame."

        obj = target_obj if target_obj and target_obj["class"] == "person" else persons[0]
        shirt = obj.get("clothing_color", "unknown")
        pants = obj.get("pants_color", "unknown")
        attrs = obj.get("attributes", {})

        details = [f"Shirt/Top: {shirt}"]
        if pants not in ("unknown", "colorful"):
            details.append(f"Pants: {pants}")
        if attrs and attrs.get("hat", {}).get("detected"):
            details.append(f"Headwear: {attrs['hat']['color']} hat")
        if attrs and attrs.get("glasses", {}).get("detected"):
            details.append("Wearing glasses")

        return f"Person (ID #{obj['id']}) outfit summary: {', '.join(details)}."

    def _handle_history_recall(self) -> str:
        if not self.conv_engine.history:
            return "No previous questions recorded in this session."
        last_turn = self.conv_engine.history[-1]
        return f"Previously, you asked: '{last_turn['query']}', and I answered: '{last_turn['answer']}'"

    # ── Standard Handlers ────────────────────────────────────────────────────

    def _describe(self, scene: dict) -> str:
        if not scene.get("objects"):
            return "No objects are currently detected in the frame."

        lines = ["Here is what I see in the frame:\n"]
        for obj in scene["objects"]:
            color_part = (
                f"{obj['color']} " if obj["color"] not in ("unknown", "colorful") else ""
            )
            posture_part = f" ({obj['posture']})" if obj.get("posture") and obj["posture"] != "unknown" else ""
            motion_part  = f" [{obj['motion_direction']}]" if obj.get("motion_direction") else ""
            lines.append(
                f"  - {color_part}{obj['class']}{posture_part}  "
                f"[Track ID #{obj['id']}]  --  {obj['position']} of frame{motion_part}"
            )

        counts = scene.get("object_counts", {})
        summary = ", ".join(f"{v} {k}(s)" for k, v in counts.items())
        lines.append(f"\nSummary: {summary}")
        return "\n".join(lines)

    def _count(self, raw_cls: str, scene: dict) -> str:
        cls = self._resolve(raw_cls)
        counts = scene.get("object_counts", {})
        cnt = counts.get(cls, 0)
        if cnt:
            return f"There are {cnt} {cls}(s) in the frame."

        for k, v in counts.items():
            if cls in k or k in cls:
                return f"There are {v} {k}(s) in the frame."

        return f"I don't see any '{raw_cls}' in the frame. " + self._visible_hint(scene)

    def _color(self, raw_obj: str, scene: dict, original_q: str, target_obj: dict | None) -> str:
        is_clothing = any(w in raw_obj for w in CLOTHING_WORDS) or any(w in original_q for w in CLOTHING_WORDS)
        cls = self._resolve(raw_obj)
        matches = self._find(cls, scene)

        if not matches:
            return f"I don't see any '{raw_obj}' in the frame. " + self._visible_hint(scene)

        if is_clothing and cls == "person":
            clothing_word = "t-shirt" if any(w in original_q or w in raw_obj for w in ["tshirt", "t-shirt", "shirt"]) else "clothing"
            obj = target_obj if target_obj and target_obj["class"] == "person" else matches[0]
            c = obj.get("clothing_color", obj["color"])
            return f"The person (ID #{obj['id']})'s {clothing_word} appears to be {c}."

        if len(matches) == 1:
            obj = matches[0]
            return f"The {obj['class']} (ID #{obj['id']}) is predominantly {obj['color']} colored."

        parts = ", ".join(f"ID #{o['id']}: {o['color']}" for o in matches)
        return f"{len(matches)} {cls}(s) detected -- colors: {parts}."

    def _where(self, raw_obj: str, scene: dict) -> str:
        cls = self._resolve(raw_obj)
        matches = self._find(cls, scene)
        if not matches:
            return f"I don't see any '{raw_obj}' in the frame. " + self._visible_hint(scene)
        if len(matches) == 1:
            obj = matches[0]
            return f"The {obj['class']} (ID #{obj['id']}) is at the {obj['position']} of the frame."
        parts = ", ".join(f"ID #{o['id']} -> {o['position']}" for o in matches)
        return f"Found {len(matches)} {cls}(s): {parts}."

    def _exists(self, raw_obj: str, scene: dict) -> str:
        cls = self._resolve(raw_obj)
        matches = self._find(cls, scene)
        if matches:
            n = len(matches)
            verb = "is" if n == 1 else "are"
            return f"Yes, there {verb} {n} {cls}(s) visible in the frame."
        return f"No, I don't see any '{raw_obj}' right now. " + self._visible_hint(scene)

    def _fallback(self, q: str, scene: dict) -> str:
        if not scene.get("objects"):
            return "No objects are detected right now. Point the camera at something and press [Q] again."
        return (
            f"I'm not sure about that specific query -- but here's what I see:\n"
            f"{scene.get('description', '')}\n\n"
            "Tier 2 capabilities you can ask:\n"
            "  - \"Does the person wear any earrings?\"\n"
            "  - \"Does the person have any weapon?\"\n"
            "  - \"Is the person sitting or standing?\"\n"
            "  - \"What is the person wearing?\"\n"
            "  - \"What color are the pants?\"\n"
            "  - \"How long has the person been here?\"\n"
            "  - \"Which direction are they moving?\"\n"
            "  - \"What is in the frame?\""
        )

    # ── Utility Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _match(text: str, keywords: list[str]) -> bool:
        for kw in keywords:
            if " " in kw:
                if kw in text:
                    return True
            else:
                if re.search(r"\b" + re.escape(kw) + r"\b", text):
                    return True
        return False

    @staticmethod
    def _extract(text: str, patterns: list[str]) -> str | None:
        for pat in patterns:
            m = re.search(pat, text)
            if m:
                return m.group(1).strip()
        return None

    @staticmethod
    def _resolve(raw: str) -> str:
        low = raw.lower().strip()
        if low in SYNONYMS:
            return SYNONYMS[low]
        key = low.rstrip("s")
        if key in SYNONYMS:
            return SYNONYMS[key]
        return key if key else low

    @staticmethod
    def _find(cls: str, scene: dict) -> list[dict]:
        out = []
        for obj in scene.get("objects", []):
            oc = obj["class"].lower()
            if cls == oc or cls in oc or oc in cls:
                out.append(obj)
        return out

    @staticmethod
    def _visible_hint(scene: dict) -> str:
        counts = scene.get("object_counts", {})
        if counts:
            visible = ", ".join(counts.keys())
            return f"Currently visible: {visible}."
        return "The frame appears to be empty."
