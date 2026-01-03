# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This repository contains the German-to-Korean translation project for "Fundamentals of Software Architecture, 2nd Edition". The content is organized into 34 chapters covering software architecture fundamentals, architecture styles, and techniques/soft skills.

### Repository Structure

The repository follows a consistent chapter-based organization:

```
{chapter_number}_{chapter_title_german}/
├── {chapter_title_german}.md          # Original German markdown
├── {chapter_title_german}_ko.md       # Korean translation
├── {chapter_title_german}.pdf         # Original PDF extract
├── {chapter_title_german}.json        # Chapter metadata
└── _page_N_Figure_*.jpeg              # Chapter images/diagrams
```

**Key directories:**
- `.claude/skills/` - Custom Claude Code skills for this project
- Each numbered directory (02_1_einfuhrung/, 20_17_orchestrierungsgesteuerte_service_orientierte_architektur/, etc.) represents a book chapter

### Content Organization

The book is divided into three main parts (see `metadata.json`):
1. **Part I: Fundamentals** (Chapters 1-8) - Core architecture concepts
2. **Part II: Architecture Styles** (Chapters 9-20) - Different architectural patterns
3. **Part III: Techniques and Soft Skills** (Chapters 21-27) - Communication and leadership

## Translation Workflow

### Using the paper-translator-korean Skill

The primary translation tool is the `paper-translator-korean` skill located in `.claude/skills/paper-translator-korean/`.

**To translate a document:**
```bash
# Use the Skill tool with the skill name
Skill: paper-translator-korean

# The skill will automatically:
# 1. Read the source German markdown file
# 2. Translate all content to Korean (preserving technical terms)
# 3. Save to {filename}_ko.md with YAML header
# 4. Keep References/Appendix sections untranslated (headers translated only)
```

**Translation conventions:**
- Technical terms: First occurrence shows both languages, e.g., "서비스 지향 아키텍처 (Service-Oriented Architecture, SOA)"
- German original terms in parentheses: "오케스트레이션 기반 (Orchestrierung)"
- All markdown formatting preserved (headers, lists, tables, code blocks, images)
- YAML header prepended for HTML rendering with embedded images

**Output file naming:**
- Input: `27_23_diagramme_zur_architektur.md`
- Output: `27_23_diagramme_zur_architektur_ko.md`

### Using the markdown-formatter Skill

The `markdown-formatter` skill in `.claude/skills/markdown-formatter/` provides document structure analysis and formatting.

**Use when:**
- Analyzing markdown document structure
- Adding headers to sections
- Emphasizing key technical terms with bold formatting
- Fixing bold formatting rendering issues in multilingual documents

**Includes Python script:**
- `scripts/fix_multilingual_bold.py` - Fixes bold rendering issues across languages

## File Structure Patterns

### Chapter Metadata (JSON files)

Each chapter directory contains a `.json` file with metadata. These are auto-generated during PDF extraction and should generally not be modified manually.

### Image References

Images follow the pattern `_page_N_Figure_*.jpeg` where N is the page number. These are referenced in markdown as:
```markdown
![](_page_2_Figure_0.jpeg)
```

### Korean Translation Files

All `*_ko.md` files should include YAML frontmatter:
```yaml
---
lang: ko
format:
  html:
    toc: true
    embed-resources: true
    theme: cosmo
---
```

This enables proper HTML rendering with embedded images when converting to other formats.

## Working with Translations

### Verifying Translation Completeness

```bash
# Count total chapters (excluding part dividers)
find . -maxdepth 1 -type d -name "*_*" | wc -l

# Count translated files
find . -name "*_ko.md" | wc -l

# Find untranslated chapters
for dir in */; do
  if [[ -f "${dir}"*.md ]] && [[ ! -f "${dir}"*_ko.md ]]; then
    echo "Untranslated: $dir"
  fi
done
```

### Quality Checks

When reviewing translations:
1. Verify technical terms follow the pattern: Korean (German/English)
2. Check that all images are referenced correctly
3. Ensure markdown structure matches the original (same header levels, list nesting)
4. Confirm YAML header is present in `*_ko.md` files

## Architecture Content Guidelines

This is a technical architecture book, so translations should:
- Preserve all technical terminology accurately
- Maintain the formal academic tone (합니다체 in Korean)
- Keep code blocks, diagrams, and examples intact
- Preserve cross-references between chapters (maintain original anchor links)

## Important Notes

- **Language**: Source content is in German, target is Korean
- **Technical terms**: Use established Korean translations where they exist (e.g., 마이크로서비스 for Microservices)
- **Citations/References**: Keep in original language, translate section headers only
- **Chapter numbering**: German chapters start at "Kapitel 1", Korean uses "1장"

## Metadata File

`metadata.json` contains the complete book structure with 34 chapters, page ranges, and file mappings. Refer to this for:
- Chapter ordering and numbering
- Original page ranges from the PDF
- Complete table of contents
