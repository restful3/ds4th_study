---
name: markdown-formatter
description: Analyze and format multilingual Markdown documents by adding headers, bold formatting to key terms, and fixing bold rendering issues across all languages (Korean, English, Chinese, Japanese, etc.). Use when the user requests Markdown document structure improvement, header organization, or bold formatting fixes.
---

# Markdown Formatter

## Overview

This skill provides comprehensive analysis and formatting capabilities for Markdown documents in any language. It automatically improves document structure, adds appropriate headers, emphasizes key terms, and fixes bold formatting rendering issues.

## When to Use This Skill

Use this skill when:
- User requests to analyze or improve Markdown document structure
- User wants to add headers to sections with numbering patterns (1., 1.1., etc.)
- User needs to emphasize key technical terms and concepts with bold formatting
- User reports bold formatting not rendering correctly in multilingual documents
- User wants to ensure consistent formatting across Markdown files

## Workflow

### Phase 1: Document Analysis

Start by thoroughly analyzing the target Markdown file:

1. **Read the entire document** using the Read tool to understand content, context, and main topics

2. **Analyze document structure** using Grep commands:
   ```bash
   # Find header structure (H1-H6)
   rg -n '^#{1,6}\s' "$TARGET_FILE"

   # Extract bold-formatted terms
   rg -n '\*\*[^*]+\*\*' "$TARGET_FILE"

   # Analyze list structure
   rg -n '^\s*[-*+]\s' "$TARGET_FILE"

   # Check code blocks
   rg -n '^```' "$TARGET_FILE"

   # Check blockquotes
   rg -n '^>' "$TARGET_FILE"
   ```

3. **Identify areas for improvement**:
   - Header hierarchy consistency and logical flow
   - Section divisions and subsections
   - Currently bold-formatted terms and their consistency
   - Overall document flow and structural completeness

### Phase 2: Header and Bold Addition

#### Adding Headers Automatically

Recognize these patterns and add appropriate Markdown headers:

- **Numbered sections**: Lines starting with `1.`, `2.`, `1.1.`, `2.1.`, etc.
  - `1.` → `## 1.` (H2 header)
  - `1.1.` → `### 1.1.` (H3 header)
  - `1.1.1.` → `#### 1.1.1.` (H4 header)
- **Standalone title lines**: Short sentences (under 20 characters) on their own line
- **Special sections**: Introduction, Summary, Conclusion, etc.

#### Adding Bold Formatting Automatically

Apply bold formatting (`**text**`) to:

**Technical Terms and Concepts:**
- English technical terms in parentheses (e.g., Microservices Architecture → 마이크로서비스 아키텍처(**Microservices Architecture**))
- Terms being defined (following "~란", "~이란", "~는")
- Important terms in quotation marks

**Key Concepts:**
- First occurrence of core concepts in the document
- 1-2 keywords representing each section's topic
- Comparison/contrast term pairs (with "vs", "반면", "대" etc.)

**Avoid over-emphasizing:**
- Prevent more than 3 bold terms per paragraph
- Don't bold common nouns
- Only bold the first occurrence in each section

**Example transformation:**

Before:
```markdown
1. 건축적 사고방식의 정의

건축적 사고방식이란 사물을 아키텍트의 눈, 즉 아키텍처의 관점에서 바라보는 것을 의미합니다.

1.1. 지식의 피라미드

모든 기술 지식은 세 가지 단계로 나눌 수 있습니다.
```

After:
```markdown
## 1. 건축적 사고방식의 정의

**건축적 사고방식**이란 사물을 아키텍트의 눈, 즉 **아키텍처의 관점**에서 바라보는 것을 의미합니다.

### 1.1. 지식의 피라미드

모든 **기술 지식**은 세 가지 단계로 나눌 수 있습니다.
```

### Phase 3: Bold Formatting Error Fixes

Fix bold rendering issues that occur across all languages due to spacing problems.

#### Common Bold Formatting Bugs

**Bug 1: Characters immediately after closing `**`**
```markdown
# Problem
**용어**한글      # Korean
**term**text      # English
**概念**데이터    # Mixed languages

# Fixed
**용어** 한글
**term** text
**概念** 데이터
```

**Bug 2: Space after opening `**`**
```markdown
# Problem
** term**
** 개념**

# Fixed
**term**
**개념**
```

**Bug 3: Special characters inside bold with characters immediately after**
```markdown
# Problem
**용어(설명)**다음      # With parentheses
**개념, 정의**texto     # With comma + Spanish
**항목.**끝             # With period + Korean
**value:**값            # With colon + Korean

# Fixed
**용어(설명)** 다음
**개념, 정의** texto
**항목.** 끝
**value:** 값
```

#### Detection Commands

First check PCRE2 support:
```bash
rg --pcre2 --version 2>&1 | grep -q "PCRE2" && echo "PCRE2 available" || echo "PCRE2 not supported"
```

**With PCRE2 support (preferred):**
```bash
# Bug 1 & 3: Characters after closing **
rg -n --pcre2 '\*\*[^*]+?\*\*(?=\p{L}|\p{N})' "$TARGET_FILE"

# Bug 2: Space after opening **
rg -n --pcre2 '(?:^|\s)\*\*\s+' "$TARGET_FILE"
```

**Without PCRE2:**
```bash
# Bug 1 & 3: Alphanumeric after closing **
rg -n '\*\*[^*]+\*\*[[:alpha:][:digit:]]' "$TARGET_FILE"

# Bug 2: Space after opening **
rg -n '\*\* ' "$TARGET_FILE"
```

#### Fixing Procedure

1. **Find all issues** using detection commands above
2. **Use Edit tool** to fix each problematic line:
   - Add space after closing `**`
   - Remove space after opening `**`
3. **Verify changes** by running detection commands again

#### Automated Fix Script

For bulk fixes or files with many issues, use the bundled script:

```bash
python3 scripts/fix_multilingual_bold.py "$TARGET_FILE"

# Verify after fixing
rg -n '\*\*[^*]+\*\*[[:alpha:][:digit:]]' "$TARGET_FILE"
```

The script automatically handles all three bug types across all languages.

### Phase 4: Validation and Reporting

#### Validation Commands

```bash
echo "=== Validating bold formatting ==="
echo ""

# Bug 1 & 3: Characters after closing **
echo "1. Characters immediately after closing **:"
rg -n '\*\*[^*]+\*\*[[:alpha:][:digit:]]' "$TARGET_FILE" && echo "  ✗ Still needs fixing" || echo "  ✓ No issues"

echo ""

# Bug 2: Space after opening **
echo "2. Space after opening **:"
rg -n '\*\* ' "$TARGET_FILE" && echo "  ✗ Still needs fixing" || echo "  ✓ No issues"

echo ""
echo "=== Validation complete ==="
```

#### Report Template

Generate a summary report with:

```markdown
## Markdown Formatting Report

### Target File
- Filename: <TARGET_FILE>
- Total lines: <count>
- Analysis date: <timestamp>

### Phase 1: Structure Analysis Results
- Headers: H1(<count>), H2(<count>), H3(<count>)...
- Bold terms: <count>
- Lists: <count>
- Code blocks: <count>

### Phase 2: Headers and Bold Addition
- Headers added: <count>
  - H2 level: <count>
  - H3 level: <count>
  - H4 level: <count>
- Bold formatting added: <count>
  - Key terms: <count>
  - Technical concepts: <count>

### Phase 3: Bold Formatting Fixes
- Bug 1 (closing ** + char/digit): <count> fixed
  - Lines: <line numbers>
- Bug 2 (opening ** + space): <count> fixed
  - Lines: <line numbers>
- Bug 3 (special chars + char): <count> fixed
  - Lines: <line numbers>

### Phase 4: Validation Results
- [x] All bold formatting errors fixed
- [x] Validation passed (no issues found)

### Total Changes
- Headers added: <total>
- Bold formatting added: <total>
- Formatting errors fixed: <total>
- **Total changes**: <sum>
```

## Best Practices

- **Backup first**: Recommend backing up original file before modifications
- **Version control**: Use Git or similar to track changes
- **Test rendering**: Verify actual display in Markdown renderer
- **Batch processing**: Use the script for multiple files

## Batch Processing Multiple Files

To process multiple Markdown files at once:

```bash
# Find all .md files and process them
find . -name "*.md" -type f | while read file; do
    if rg -q '[가-힣一-龥A-Za-z]' "$file"; then
        echo "Processing: $file"
        python3 scripts/fix_multilingual_bold.py "$file"
    fi
done
```

## Troubleshooting

**Q1: PCRE2 not supported**

Update ripgrep to latest version:
```bash
# Ubuntu/Debian
sudo apt update && sudo apt install ripgrep

# macOS
brew install ripgrep
```

**Q2: Python script execution error**

Check Python 3 installation:
```bash
python3 --version
chmod +x scripts/fix_multilingual_bold.py
```

**Q3: Some bold formatting still broken**

1. Check for special character combinations (emojis, special Unicode)
2. Verify Markdown renderer compatibility
3. Manually review and fix specific cases

**Q4: Unwanted modifications**

1. Restore from Git: `git checkout -- <file>`
2. Restore from backup
3. Add exclusion patterns to script if needed

## Technical Notes

- **Markdown dialect**: Based on CommonMark and GitHub Flavored Markdown
- **ripgrep version**: 13.0+ recommended (for PCRE2 support)
- **Python version**: Python 3.6+ required
- **Encoding**: All files assumed UTF-8
- **Performance**: Use script for large files (1000+ lines)