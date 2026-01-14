# Web to Markdown Conversion Prompt Guide

This guide describes how to request a high-fidelity conversion of a web page into a Markdown document, ensuring charts (Mermaid), tables, and full text are preserved exactly as they appear.

## Recommended Prompt Template

Copy and paste the following prompt when you want to convert a web page to Markdown. Replace `[URL]` with the target page URL.

```markdown
I need you to convert the following web page into a high-fidelity Markdown file: [URL]

Please use a browser subagent to perform this task with the following strict adherence instructions:

1. **Goal**: Capture the COMPLETE page content word-for-word. Do NOT summarize, paraphrase, or omit any details.
2. **Format**:
   - Return the **FULL RAW MARKDOWN** text in your final report so I can copy-paste it directly.
   - Use standard Markdown syntax (headers `##`, lists `-`, bold `**`, etc.).
3. **Diagrams**:
   - Inspect any diagrams or flowcharts visually.
   - Convert them into **Mermaid** graphs (`mermaid` code blocks) that accurately represent the nodes and connections.
4. **Tables**:
   - formatting of tables must be exact. Transfer all rows and columns into Markdown tables.
5. **Content**:
   - Capture ALL sections, including "Notes", "Warnings", "Callouts".
   - Capture ALL code blocks exactly as written.
   - Do not skip detailed explanations or lists.

**CRITICAL**: The final output must be the Markdown content itself, not a description of what you saw.
```

## Strategy Explanation

To achieve high-quality results, the prompt enforces the following strategies:

1.  **Explicit Format Request**: By asking for "FULL RAW MARKDOWN", the agent is instructed to generate the code representation directly, rather than just describing the page.
2.  **Anti-Summarization**: Commands like "Word-for-word", "Do NOT summarize" prevent the LLM's natural tendency to condense information.
3.  **Visual Translation**: Explicitly instructing to convert diagrams to **Mermaid** ensures visual elements are preserved in a code-friendly format.
4.  **Subagent Delegation**: Requesting a "browser subagent" ensures a specialized task runner is used to render the page, handle scrolling, and extract dynamic content effectively.
