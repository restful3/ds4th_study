# Web to Markdown Conversion Guide

This guide outlines the **"Zero Information Loss"** standard for converting technical documentation from the web into high-fidelity Markdown. The goal is to create a local text-based replica that is indistinguishable in content from the original web page.

## Core Philosophy: Zero Information Loss

1.  **Word-for-Word Fidelity**: Never summarize. Never paraphrase. If the original text says "Furthermore, it is important to note that...", the Markdown must say exactly that, not "Note that...".
2.  **Complete Structure**: Headings, sub-headings, notes, warnings, and sidebars must be preserved with appropriate Markdown syntax (e.g., `> [!NOTE]`).
3.  **Visuals as-is**: Images should be placed exactly where they appear in the flow. Diagrams should be converted to Mermaid if possible, or kept as images if complex.

## Handling Complex UI Patterns

### 1. Tabbed Content (The "Sequential Expansion" Strategy)

Technical documentation often uses tabs to switch between languages (e.g., "Python" vs "JavaScript") or APIs (e.g., "Graph API" vs "Functional API").

**Rule**: Do not pick just one. Extract **ALL** tabs and present them sequentially.

**Example Scenario**:
A page has a code block with two tabs: **Graph API** and **Functional API**.

**Web UI**:
```
[ Graph API ] [ Functional API ]
< content visible based on click >
```

**Markdown Output**:
```markdown
#### Graph API

```python
# Code for Graph API...
```

#### Functional API

```python
# Code for Functional API...
```
```

### 2. Images & Media

*   **Extraction**: Download images to a local `images/` directory relative to the Markdown file.
*   **Naming**: Use descriptive names matching the content (e.g., `checkpoints_full_story.avif` instead of `img_01.png`).
*   **Placement**: Insert the image syntax `![Alt Text](images/filename.ext)` exactly at the logical break point where it sits in the web layout.

## Verification Checklist

Before considering a conversion complete, verify:

- [ ] **The "Bottom of Page" Trap**: Did you capture the very last section? (e.g., "Capabilities" lists often have items at the footer).
- [ ] **Tab Expansion**: Are both "Option A" and "Option B" code blocks present?
- [ ] **Image Count**: If the web page has 5 images, does the Markdown have 5 image tags?
- [ ] **Code Accuracy**: Are comments, decorators, and imports in code blocks preserved exactly?

## Recommended Prompt Template

Use this prompt to instruct the agent:

```markdown
I need to convert [URL] into a high-fidelity Markdown file.

**Strict Requirements:**
1. **No Summarization**: Capture text word-for-word.
2. **Tabbed Content**: If there are tabs (e.g., Graph API vs Functional API), extract content for BOTH and list them sequentially under h4 headers.
3. **Images**: Use existing local images from `images/` directory. Place them exactly as they appear in the source.
4. **Verification**: Double-check the bottom of the page to ensure the final sections (e.g., "Next Steps", "Capabilities") are not cut off.

Use the `browser_subagent` to view the page and extract exact content.
```
