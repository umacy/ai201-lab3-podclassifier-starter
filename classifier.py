import json
import os
from groq import Groq
from config import GROQ_API_KEY, LLM_MODEL, VALID_LABELS, DATA_PATH, TRAIN_FILE, LABELS_FILE

_client = Groq(api_key=GROQ_API_KEY)


def load_labeled_examples() -> list[dict]:
    """
    Load the training episodes and merge them with the student's labels.

    Returns a list of dicts, each with:
      - "id"          : episode ID
      - "title"       : episode title
      - "podcast"     : podcast name
      - "description" : episode description
      - "label"       : the label from my_labels.json (may be None if not yet annotated)

    Only returns episodes where the label is a valid, non-null string.
    Episodes with null labels are silently skipped.
    """
    train_path = os.path.join(DATA_PATH, TRAIN_FILE)
    labels_path = os.path.join(DATA_PATH, LABELS_FILE)

    with open(train_path, encoding="utf-8") as f:
        episodes = {ep["id"]: ep for ep in json.load(f)}

    with open(labels_path, encoding="utf-8") as f:
        labels = {entry["id"]: entry["label"] for entry in json.load(f)}

    labeled = []
    for ep_id, ep in episodes.items():
        label = labels.get(ep_id)
        if label in VALID_LABELS:
            labeled.append({**ep, "label": label})

    return labeled


def build_few_shot_prompt(labeled_examples: list[dict], description: str) -> str:
    """
    Build a few-shot classification prompt using the student's labeled training examples.

    Structure (see specs/classifier-spec.md):
      1. Task instruction + the four label definitions.
      2. The labeled examples, one block each (Title / Description / Label).
         Omitted entirely if labeled_examples is empty (zero-shot fallback).
      3. The new episode in the same block format with "Label: ?", followed by
         an explicit two-line output-format instruction the parser relies on.
    """
    instruction = (
        "You are classifying podcast episodes by their format. "
        "Classify the episode into exactly one of these four labels:\n\n"
        "- interview: a conversation between a host and one or more guests\n"
        "- solo: a single host speaking from memory, experience, or opinion "
        "— no guests, no assembled external sources\n"
        "- panel: multiple guests with roughly equal speaking time, often "
        "debating or discussing a topic together\n"
        "- narrative: a story assembled from external sources — interviews, "
        "archival audio, reporting — with a clear narrative arc\n\n"
        "Return only the label and your reasoning. Do not explain the taxonomy."
    )

    parts = [instruction]

    if labeled_examples:
        example_blocks = []
        for ex in labeled_examples:
            example_blocks.append(
                f"Title: {ex['title']}\n"
                f"Description: {ex['description']}\n"
                f"Label: {ex['label']}"
            )
        parts.append("Here are labeled examples:\n\n" + "\n\n---\n\n".join(example_blocks))

    parts.append(
        "Now classify this episode:\n\n"
        f"Description: {description}\n"
        "Label: ?\n\n"
        "Respond with exactly two lines, in this format, and nothing else:\n"
        "Label: <interview | solo | panel | narrative>\n"
        "Reasoning: <one or two sentences>"
    )

    return "\n\n".join(parts)


def _parse_response(text: str) -> dict:
    """
    Extract a label and reasoning from the LLM's text response.

    Parses by line PREFIX, not position, so a conversational preamble does not
    throw off label extraction. Normalizes the label (lower-case, strip
    whitespace and surrounding markdown/punctuation) before validating it
    against VALID_LABELS; anything that does not match becomes "unknown".
    """
    label = None
    reasoning = ""

    for line in text.splitlines():
        # Strip leading markdown (e.g. "**Label:**", "- Label:") before the
        # prefix check so formatted and plain lines match the same way.
        stripped = line.strip().lstrip("*#-> `").strip()
        lowered = stripped.lower()
        if label is None and lowered.startswith("label:"):
            raw = stripped.split(":", 1)[1]
            label = raw.strip().strip("*#`. ").lower()
        elif not reasoning and lowered.startswith("reasoning:"):
            reasoning = stripped.split(":", 1)[1].strip().strip("*` ")

    # Fallback: no "Label:" line — scan for the first whole-word valid label.
    if label not in VALID_LABELS:
        words = text.lower().replace("\n", " ").split()
        cleaned = [w.strip("*#`.,:;()[]\"'") for w in words]
        for valid in VALID_LABELS:
            if valid in cleaned:
                label = valid
                break

    if label not in VALID_LABELS:
        label = "unknown"

    if not reasoning:
        reasoning = text.strip()

    return {"label": label, "reasoning": reasoning}


def classify_episode(description: str, labeled_examples: list[dict]) -> dict:
    """
    Classify a single podcast episode description using the few-shot LLM classifier.

    Returns a dict with "label" (one of VALID_LABELS, or "unknown") and
    "reasoning". Never raises and never returns None — any API or parsing
    failure degrades to {"label": "unknown", ...} so the 20-call evaluation
    loop in Milestone 3 keeps running on a single bad response.
    """
    try:
        prompt = build_few_shot_prompt(labeled_examples, description)

        response = _client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
        )

        text = response.choices[0].message.content
        if not text:
            return {
                "label": "unknown",
                "reasoning": "Empty response from the LLM.",
            }

        return _parse_response(text)

    except Exception as e:
        return {
            "label": "unknown",
            "reasoning": f"Error: {e}",
        }
