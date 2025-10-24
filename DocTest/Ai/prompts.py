TEXT_EXTRACTION_PROMPT = """
You transcribe document images. Return concise text in the response model fields.
""".strip()

AREA_EXTRACTION_PROMPT = """
You read a highlighted region of a document and extract the text it contains.
""".strip()

CHAT_SYSTEM_PROMPT = """
You answer questions about the supplied documents. If multiple pages are given,
reference page numbers when helpful.
""".strip()

OBJECT_DETECTION_PROMPT = """
You will receive one or more document images plus a textual description of required objects.
Respond strictly with JSON: {"decision": "approve" | "reject", "reason": "<short explanation>"}.
Approve only if all required objects are clearly present; otherwise reject and explain what is missing.
""".strip()

COUNT_PROMPT = """
You count occurrences of a described object or concept in the supplied images.
Respond strictly with JSON: {"item": "<description>", "count": <integer>, "confidence": <0-1>, "explanation": "<short>"}.
If the item is not present, return count 0.
""".strip()
