---
name: paper-translator-korean
description: Translate academic papers and technical documents in markdown format from any foreign language to Korean. Use when user requests translation of research papers, academic documents, technical books, or scientific articles to Korean.
---

# Paper Translator (Any Language to Korean)

## When to Use
Use this skill when:
- User requests translation of academic papers or technical documents to Korean
- Input is markdown-formatted academic content in any language (English, German, French, Spanish, Japanese, Chinese, etc.)
- User asks to translate research papers, journal articles, technical books, or scientific documents
- Source language can be any foreign language - the skill will automatically detect and translate to Korean

**Default behavior: Full paper translation in one go**
- Translate the entire paper automatically without user approval
- Process all sections sequentially without asking for confirmation
- DO NOT summarize - translate every single word and sentence completely

## Translation Guidelines

### 1. Pre-processing: Clean OCR Artifacts
**IMPORTANT**: Papers converted from OCR may contain noise that should be ignored:

- **Meaningless line breaks**: Remove unnecessary line breaks within paragraphs
  - Example: "This is a\nsentence." → "This is a sentence."
  - Keep intentional paragraph breaks
- **Header/Footer artifacts**: Ignore and remove:
  - Page numbers (e.g., "Page 5", "5", "- 5 -")
  - Journal names repeated on every page
  - Author names in headers
  - Copyright notices (e.g., "© 2023 IEEE")
  - Conference names (e.g., "NeurIPS 2023")
  - DOI strings at top/bottom
  - Running headers/footers
- **Hyphenation errors**: Merge words split across lines
  - Example: "compu-\nter" → "computer"
- **Extra whitespace**: Normalize spacing between words and paragraphs
- **OCR misreads**: Use context to correct obvious errors if possible

**Clean first, then translate**: Remove all OCR noise before beginning translation.

### 2. Analyze the Paper Structure
- **Auto-detect source language**: The document can be in any language (English, German, French, Spanish, Japanese, Chinese, etc.)
- Identify sections (Abstract/Zusammenfassung/要約, Introduction/Einleitung/序論, Methods, Results, Discussion, Conclusion, etc.)
- **IMPORTANT**: Identify sections that should NOT be translated:
  - **References/Bibliography** (REFERENCES/Literatur/参考文献, Bibliography, Works Cited, etc.)
  - **Appendix/Appendices** (APPENDIX/Anhang/付録, Supplementary Material, etc.)
  - **Acknowledgements/Acknowledgments** (ACKNOWLEDGEMENTS/Danksagung/謝辞, Acknowledgment, etc.)
  - These sections should be copied as-is without translation
- Note technical terms, equations, citations, and figures (in any language)
- Preserve markdown formatting (headers, lists, tables, code blocks)

### 3. Translation Strategy

**CRITICAL - Complete Translation Without User Approval:**
- **Translate the entire paper automatically** from start to finish
- **NO summarization** - translate every single word and sentence completely
- **NO user confirmation** - process all sections sequentially without asking
- Start with Abstract and continue through all sections automatically
- Only stop at sections that should NOT be translated (References, Appendix, Acknowledgements)

**Automatic Full-Paper Approach:**
1. Analyze the paper structure silently (no user prompts)
2. Start translating from Abstract automatically
3. Continue through all sections without stopping for approval
4. Save all translated sections progressively to the output file
5. Complete the entire translation in one continuous workflow

**Translation Guidelines per Section:**
- **Complete Translation**: Translate every single sentence - never summarize or skip content
- **Academic Terms**: Maintain original language terms in parentheses for key technical vocabulary
  - Example (English): "machine learning (머신러닝)" on first occurrence
  - Example (German): "Softwarearchitektur (소프트웨어 아키텍처)" on first occurrence
  - Example (Japanese): "機械学習 (머신러닝/기계학습)" on first occurrence
- **Maintain Formality**: Use formal Korean academic writing style (합니다체)
- **Preserve Citations**: Keep citation formats unchanged [1], (Smith et al., 2023), etc.
- **Equations**: Keep mathematical notation as-is
- **Figures/Tables**: Translate captions and descriptions to Korean

### 4. Step-by-Step Translation Process

**CRITICAL: Automatic full-paper translation without user approval!**

1. **Clean OCR artifacts** (FIRST STEP):
   - Remove page numbers, headers, footers
   - Fix meaningless line breaks within sentences/paragraphs
   - Merge hyphenated words split across lines
   - Remove repeated journal/conference information

2. **Analyze structure silently**:
   - Identify all sections in the paper
   - Note sections to skip (References, Appendix, Acknowledgements)
   - DO NOT ask user for confirmation - proceed automatically

3. **Translate ALL sections automatically**:
   - Start from Abstract (or equivalent in source language)
   - Continue through all sections sequentially without asking
   - **SKIP sections that should NOT be translated**:
     - References/Bibliography: Translate section header to "## 참고문헌 (REFERENCES/Literatur/参考文献)" but keep all reference entries in original language
     - Appendix: Translate section header to "## 부록 (APPENDIX/Anhang/付録)" but keep content in original language
     - Acknowledgements: Translate section header to "## 감사의 글 (ACKNOWLEDGEMENTS/Danksagung/謝辞)" but keep content in original language
   - **NEVER ask user for approval between sections**

4. **For each section translation**:
   - **Complete translation**: Translate every single word and sentence - no summarization
   - Maintain section header format
   - For technical terms:
     - First occurrence: "English term (한국어 번역)"
     - Subsequent: Use Korean or English based on common usage
   - Preserve all markdown formatting:
     - Headers (#, ##, ###)
     - Lists (-, *, 1.)
     - Code blocks (```)
     - Tables (|)
     - Links and references

5. **Save progressively**:
   - Save translated content as each section completes
   - Continue to next section automatically
   - Report completion only when entire paper is done

**Example workflow:**
- User: "이 논문을 번역해줘" (또는 "ml_survey.md를 번역해줘")
- Claude: [Silently analyzes structure] → [Translates Abstract completely] → [Saves to ml_survey_ko.md with YAML header] → [Translates Introduction completely] → [Appends to ml_survey_ko.md] → [Continues through all sections] → "전체 논문 번역이 완료되었습니다. ml_survey_ko.md 파일에 저장했습니다."

### 4. Quality Checks
- Ensure natural Korean sentence flow
- Verify technical accuracy
- Check that all sections are translated
- Confirm markdown structure is preserved

### 5. File Saving Protocol

**CRITICAL: Always save translated sections to a file with YAML header**

**File naming convention:**
- Original file: `paper.md` or `research_paper.md`
- Translated file: `paper_ko.md` or `research_paper_ko.md`
- Pattern: `{original_filename}_ko.md`

**YAML Header Requirement:**
- **ALWAYS prepend YAML header from `header.yaml` before any content**
- This is critical for proper HTML rendering with embedded images
- The header contains `embed-resources: true` for image embedding
- Header path: `/media/restful3/data/workspace/paperflow/header.yaml` or `header.yaml` in project root

**Saving behavior:**
1. **First section (Abstract)**: Create new file with YAML header
   - Read `header.yaml` from project root
   - Write YAML header to new `{filename}_ko.md` file
   - Append translated Abstract section
   - **Do NOT write YAML header again in subsequent sections**

2. **Subsequent sections**: Append only the translated content
   - Append to existing `{filename}_ko.md` (which already has YAML header)
   - Do NOT add YAML header again

3. **File location**: Save in the same directory as the original file

4. **Example workflow:**
   ```
   Original: /papers/ml_survey.md

   Step 1: Translate Abstract
   → Create /papers/ml_survey_ko.md with:
      1. YAML header from header.yaml
      2. Translated Abstract content

   Step 2: Translate Introduction
   → Append Introduction to /papers/ml_survey_ko.md (no YAML header)

   Step 3: Translate Methods
   → Append Methods to /papers/ml_survey_ko.md (no YAML header)

   Final result: Complete translated paper in ml_survey_ko.md with YAML header
   ```

5. **After each section translation**:
   - Save/append the section to `{filename}_ko.md` silently
   - Continue to next section automatically without asking
   - Only report when entire paper is complete

6. **If no original filename provided**:
   - Ask user: "번역 결과를 저장할 파일명을 알려주세요 (예: paper_ko.md)"
   - Or use default: `translated_paper_ko.md`

7. **If header.yaml is not found**:
   - Use a minimal fallback header with `embed-resources: true`
   - Warn the user: "header.yaml을 찾을 수 없어 기본 헤더를 사용했습니다."

## Examples

### Example 1: OCR Noise Cleaning

**Input (with OCR artifacts):**
```markdown
NeurIPS 2023                                                    Page 5

## Abstract
Machine learning has revolu-
tionized data analysis...

© 2023 Neural Information Processing Systems
```

**After Cleaning:**
```markdown
## Abstract
Machine learning has revolutionized data analysis...
```

### Example 2: Translation with File Saving (with YAML Header)

**Scenario:** User has `deep_learning_survey.md`

**Claude's workflow:**
```
1. Analyze structure
   → Sections: Abstract, Introduction, Related Work, Methods

2. Translate Abstract (FIRST SECTION - ADD YAML HEADER)
   → Read header.yaml from project root
   → Create deep_learning_survey_ko.md with:
   ---
   lang: ko
   format:
     html:
       toc: true
       embed-resources: true
       theme: cosmo
   ---

   ## 초록 (Abstract)
   딥러닝(deep learning)은...

3. Translate Introduction (SUBSEQUENT SECTION - NO YAML HEADER)
   → Append to deep_learning_survey_ko.md:
   ## 서론 (Introduction)
   최근 몇 년간...

4. Continue for each section
   → Final file: deep_learning_survey_ko.md (complete translation with YAML header)
```

### Example 3: Translation with Technical Terms

**Input (English):**
```markdown
## Abstract
Machine learning has revolutionized data analysis...

## Introduction
Recent advances in deep learning...
```

**Output (Korean) saved to `{filename}_ko.md`:**
```markdown
## 초록 (Abstract)
머신러닝(machine learning)은 데이터 분석에 혁명을 가져왔습니다...

## 서론 (Introduction)
최근 딥러닝(deep learning)의 발전은...
```

**Input (German):**
```markdown
## Kapitel 4. Architektonische Merkmale definiert
Softwarearchitekten müssen architektonische Merkmale definieren...

## Architektonische Merkmale und Systemdesign
Um als Architekturmerkmal zu gelten, muss eine Anforderung drei Kriterien erfüllen...
```

**Output (Korean) saved to `{filename}_ko.md`:**
```markdown
## 4장. 아키텍처 특성 정의 (Kapitel 4. Architektonische Merkmale definiert)
소프트웨어 아키텍트는 아키텍처 특성(architektonische Merkmale)을 정의해야 합니다...

## 아키텍처 특성과 시스템 설계 (Architektonische Merkmale und Systemdesign)
아키텍처 특성으로 간주되려면 요구사항이 세 가지 기준을 충족해야 합니다...
```

### Example 4: Handling References Section

**Input (English):**
```markdown
## VI. CONCLUSION
This paper reviewed major AI frameworks...

## REFERENCES

[1] J. Smith, "Machine Learning Basics," IEEE Trans., 2023.
[2] A. Jones, "Deep Learning Systems," ACM Conf., 2024.
```

**Output (Korean) - References kept in English:**
```markdown
## VI. 결론 (CONCLUSION)
본 논문은 주요 AI 프레임워크를 검토했습니다...

## 참고문헌 (REFERENCES)

[1] J. Smith, "Machine Learning Basics," IEEE Trans., 2023.
[2] A. Jones, "Deep Learning Systems," ACM Conf., 2024.
```

## Special Considerations

### OCR Artifact Patterns to Remove
Common patterns found in OCR-processed papers:
- Page numbers: `Page 1`, `- 1 -`, `1 of 10`, standalone numbers at top/bottom
- Running headers: Author names, paper titles repeated on each page
- Journal info: `IEEE Transactions on...`, `Proceedings of...`
- Copyright: `© 2023`, `All rights reserved`
- DOI: `DOI: 10.xxxx/xxxxx` at top or bottom
- Timestamps: `Received: Jan 2023, Accepted: Mar 2023`
- Hyphenation: `compu-\nter`, `analy-\nsis`, `informa-\ntion`

### Content Preservation
- For mathematical papers: Keep all LaTeX/equations unchanged
- For CS papers: Programming terms often stay in original language (English, etc.)
- For medical papers: Use established Korean medical terminology
- For multilingual papers: Preserve technical terms in their original language when appropriate
- Maintain all reference numbers and bibliography format (regardless of source language)

## Common Technical Terms Guide

### Computer Science & AI
- Machine Learning → 머신러닝
- Deep Learning → 딥러닝
- Neural Network → 신경망
- Algorithm → 알고리즘
- Dataset → 데이터셋
- Model → 모델
- Training → 학습
- Validation → 검증
- Testing → 테스트
- Accuracy → 정확도
- Precision → 정밀도
- Recall → 재현율

### Research Methodology
- Methodology → 방법론
- Experiment → 실험
- Analysis → 분석
- Results → 결과
- Discussion → 논의
- Conclusion → 결론
- Hypothesis → 가설
- Variable → 변수
- Correlation → 상관관계
- Statistical Significance → 통계적 유의성

## Translation Best Practices

1. **Complete translation without summarization** (MOST IMPORTANT): Translate every single word and sentence
2. **Automatic workflow**: Process entire paper without user approval between sections
3. **First sentence of each section**: Translate completely, setting the tone
4. **Technical accuracy over literal translation**: Prioritize meaning
5. **Consistency**: Use the same translation for repeated terms
6. **Natural flow**: Korean sentence structure may differ from English
7. **Academic conventions**: Follow Korean academic writing norms

## Handling Token Limits

**Error: "Claude's response exceeded the 32000 output token maximum"**

This means the paper is too long to translate in one go. Solution:
1. **Always use section-by-section approach** (see Step-by-Step Process above)
2. If a single section is too long, split into subsections:
   - "Introduction"을 3.1, 3.2, 3.3으로 나눠서 번역
3. Create separate output files for each section if needed
4. User can combine sections later

**Recommended approach for very long papers:**
```
논문 구조:
- Abstract (짧음 - 한번에 번역)
- Introduction (중간 - 한번에 또는 2개 파트로)
- Related Work (긺 - 3-4개 파트로 나눔)
- Methods (매우 긺 - 5-6개 파트로 나눔)
...

각 파트를 순차적으로 번역하되, 사용자 승인 없이 자동으로 진행
```

## Formatting Preservation Checklist
- [ ] OCR artifacts removed (page numbers, headers, footers)
- [ ] Line breaks normalized (meaningless breaks removed)
- [ ] Hyphenated words merged
- [ ] All headers maintained with same level
- [ ] Lists formatted correctly
- [ ] Code blocks preserved with syntax highlighting
- [ ] Tables aligned properly
- [ ] Links functional
- [ ] Citations in original format
- [ ] Equations/formulas unchanged
- [ ] Image references intact
- [ ] **YAML header from `header.yaml` prepended to first section**
- [ ] **YAML header contains `embed-resources: true` for image embedding**
- [ ] **Translated content saved to `{filename}_ko.md`**
- [ ] **Each section appended to the same output file**
- [ ] **No duplicate YAML headers in subsequent sections**
- [ ] **References/Bibliography section: Header translated, content kept in English**
- [ ] **Appendix section: Header translated, content kept in English**
