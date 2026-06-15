# Classifier Spec — Pod Classifier

Complete this spec **before** writing any code for Milestone 2.

Use Plan or Ask mode to think through each blank field. When you're done,
your answers here become the blueprint for `build_few_shot_prompt()` and
`classify_episode()` in `classifier.py`.

---

## build_few_shot_prompt(labeled_examples, description)

### What it does
Constructs a prompt string for the LLM that includes the task instructions,
all labeled training examples, and the new episode description to classify.

### Inputs

| Parameter | Type | Description |
|---|---|---|
| `labeled_examples` | `list[dict]` | Each dict has `"title"`, `"description"`, `"label"` (and others). These are the examples you labeled in Milestone 1. |
| `description` | `str` | The episode description to classify. |

### Output

| Return value | Type | Description |
|---|---|---|
| prompt | `str` | A complete prompt string ready to send to the LLM. |

---

### Spec fields — fill these in before writing code

**Task instruction (what should the LLM know about the task?):**

```
You are classifying podcast episodes by their format. Classify the episode
into exactly one of these four labels:

- interview: a conversation between a host and one or more guests
- solo: a single host speaking from memory, experience, or opinion — no guests,
  no assembled external sources
- panel: multiple guests with roughly equal speaking time, often debating or
  discussing a topic together
- narrative: a story assembled from external sources — interviews, archival
  audio, reporting — with a clear narrative arc

Return only the label and your reasoning. Do not explain the taxonomy.
```

---

**How should labeled examples be formatted in the prompt?**

```
Each example should include the episode title, a brief excerpt or the full
description, and the correct label. Separate examples with a blank line or
a delimiter like "---". Include all fields that help the model see why the
label was applied — title and description are both useful; other fields
(like episode ID) are not needed.
```

---

**Example block sketch (write one concrete example):**

```
Title: {title}
Description: {description}
Label: {label}
```

---

**How should the new episode (to be classified) be presented?**

```
Present it in the same format as the labeled examples, but omit the Label
line and replace it with an instruction to classify. For example:

Title: {title}
Description: {description}
Label: ?

Then add a line like: "Classify the episode above. Return your answer in
the format below:" followed by the output format you chose.
```

---

**What output format should you request from the LLM?**

```
A two-line, prefix-labeled format:

    Label: <one of interview | solo | panel | narrative>
    Reasoning: <one or two sentences>

I considered three options:

  1. Bare label on its own line — trivial to parse, but throws away the
     reasoning the output contract requires. Rejected.
  2. JSON ({"label": ..., "reasoning": ...}) — clean if it parses, but
     llama-3.3-70b frequently wraps JSON in ```json fences or adds a
     sentence of preamble ("Here is the classification:"), which breaks
     json.loads() and forces fence-stripping/regex anyway. More failure
     modes, not fewer.
  3. Prefix-labeled lines (chosen) — robust to extra whitespace, casing,
     and stray prose. I parse by scanning lines for the "Label:" and
     "Reasoning:" prefixes (case-insensitive) and taking the text after
     the colon. Degrades gracefully: if the model omits the Reasoning
     line I still recover the label; if it adds a preamble line I skip it.

The format must match the example/output instruction in the prompt exactly
so the model mirrors it. I will explicitly instruct: "Respond with exactly
two lines, in this format, and nothing else."
```

---

**Edge cases to handle in the prompt:**

```
- labeled_examples is empty: the prompt still works as a zero-shot prompt.
  The task instruction already defines all four labels in plain language,
  so the model can classify without examples — accuracy will be lower, but
  it won't crash or produce a malformed prompt. I build the examples block
  only if the list is non-empty; otherwise I omit that section entirely
  (no dangling "Examples:" header with nothing under it).

- Very short / empty description: I still present it under the same
  Title/Description format and ask for a classification. A near-empty
  description gives the model little to work with, so it will likely pick
  the closest label or hedge — that's fine. classify_episode() validates
  the returned label against VALID_LABELS and falls back to "unknown" if
  the model can't commit, so a thin description can never crash parsing.

- The new description could itself contain the strings "Label:" or
  "Reasoning:". That only matters when parsing the RESPONSE, not the
  prompt, so it doesn't affect prompt construction — but it's why the
  parser scans the model's reply lines rather than the whole blob.
```

---

## classify_episode(description, labeled_examples)

### What it does
Classifies a single podcast episode description using the few-shot LLM classifier.
Returns a dict with a label and reasoning.

### Inputs

| Parameter | Type | Description |
|---|---|---|
| `description` | `str` | The episode description to classify. |
| `labeled_examples` | `list[dict]` | Labeled training examples from `load_labeled_examples()`. |

### Output

| Return value | Type | Description |
|---|---|---|
| result | `dict` | Must have keys `"label"` and `"reasoning"`. `"label"` must be one of `VALID_LABELS` or `"unknown"`. |

---

### Spec fields — fill these in before writing code

**Step 1 — Build the prompt:**

```
Call build_few_shot_prompt(labeled_examples, description) and store the
returned string in a variable (e.g., prompt). Pass through both arguments
exactly as received — no modification needed before calling.
```

---

**Step 2 — Send to the LLM:**

```
Call _client.chat.completions.create() with:
  - model: the model name from config (LLM_MODEL)
  - messages: a list with one dict — {"role": "user", "content": prompt}
    (system-design.md shows an optional system message too — either shape works)
  - max_tokens: a reasonable limit (e.g., 200–300) to keep responses concise

Extract the response text from:
  response.choices[0].message.content
```

---

**Step 3 — Parse the response:**

```
The chosen format is two prefix-labeled lines:
    Label: <one of the four>
    Reasoning: <a sentence or two>

Parse by PREFIX, not position (the model may add a preamble line):

  1. Split response.choices[0].message.content into lines.
  2. For the label: find the first line whose stripped, lower-cased text
     starts with "label:". Take the text after the colon, then normalize it:
     lower-case, strip whitespace, and strip surrounding markdown/punctuation
     (* # ` and a trailing period). Call this raw_label.
  3. For the reasoning: find the first line starting with "reasoning:" and
     take the text after the colon, stripped. If absent, leave reasoning as
     an empty string (or fall back to the whole response text) — a missing
     reasoning line must NOT prevent us from recovering the label.
  4. Fallback if no "label:" line exists: scan the whole response for the
     first whole-word, case-insensitive occurrence of one of VALID_LABELS
     and use that as raw_label. This rescues replies that ignored the format
     but still named a valid label.

Why prefix-based: it tolerates conversational preamble ("Sure, here's the
classification:"), stray markdown, and casing/whitespace drift, and it
decouples label extraction from reasoning so a malformed reasoning line
never costs us the label.
```

---

**Step 4 — Validate the label:**

```
Snap the parsed raw_label to the fixed set:

  - If raw_label is exactly one of VALID_LABELS (after the Step 3
    normalization), use it as-is.
  - Otherwise set label = "unknown".

This is a strict membership check against config.VALID_LABELS — we never
invent a label or "best-guess" a misspelling into a real class, because a
silently wrong label is worse for evaluation than an honest "unknown".
"unknown" is intentionally NOT in VALID_LABELS, so it shows up distinctly
in the evaluation report (it can never count as a correct prediction).

The reasoning is kept regardless of whether the label validated — even an
"unknown" result carries whatever explanation the model gave, which is
useful when debugging why a parse or classification failed.
```

---

**Step 5 — Handle errors gracefully:**

```
Wrap the API call + parsing in a try/except so one bad episode can never
crash the 20-call evaluation loop.

What can go wrong, and the response:

  - Network / API error (timeout, rate limit, auth failure, 5xx from Groq),
    or the SDK raising: catch the exception and return
      {"label": "unknown", "reasoning": "Error: <exception message>"}
  - Empty or None response content (no choices, content is None): treat as
    unparseable — return label "unknown" with a short note in reasoning.
  - Unparseable text (no "label:" line AND no whole-word label match in the
    fallback): label stays "unknown", reasoning holds the raw response so
    we can see what the model actually said.

In all failure paths the function still returns a well-formed dict with the
required "label" and "reasoning" keys and a label of "unknown" — never None,
never a raised exception. The evaluation loop counts "unknown" as a miss and
keeps going, so partial failures degrade the accuracy score honestly instead
of aborting the run.
```

---

### Return value structure

```python
{
    "label": str,      # one of VALID_LABELS, or "unknown" if invalid/error
    "reasoning": str,  # brief explanation from the LLM
}
```

---

## Notes on label quality

The classifier is only as good as your labels. If your training examples have
inconsistent or ambiguous labels, the LLM will learn the wrong pattern.

Before implementing the classifier, re-read `data/taxonomy.md` and double-check
any labels you're unsure about. Annotation quality is part of the lab.

---

## Implementation Notes

*Fill this in after implementing and testing both functions.*

**Test: what does the raw LLM response look like for one episode?**

```
Episode tested: [title]
Raw response text: [paste it here]
```

**How did you parse the label out of the response?**

```
[describe the string operations — strip, split, lower, etc.]
```

**Did any episodes return `"unknown"`? If so, why?**

```
[yes / no — if yes, what did the raw response look like?]
```

**One thing about the output format that surprised you:**

```
[your answer here]
```
