#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
다국어 Markdown 볼드 형식 자동 수정 스크립트
"""

import re
import sys
import unicodedata


def _needs_space(char: str) -> bool:
    """한 글자에 대해 공백 삽입 여부를 결정합니다."""
    if not char:
        return False
    category = unicodedata.category(char)
    return category.startswith('L') or category.startswith('N')


def fix_multilingual_bold(content: str) -> str:
    """
    다국어 Markdown의 볼드 형식 오류를 수정합니다.

    수정 사항:
    1. `**term**text` → `**term** text`
    2. `** 용어**` → `**용어**`
    3. `**value(설명)**값` → `**value(설명)** 값`
    """

    pattern = re.compile(r'\*\*([^*]+)\*\*(\S)')

    def _insert_space(match: re.Match) -> str:
        bold_text = match.group(1)
        next_char = match.group(2)
        if _needs_space(next_char):
            return f"**{bold_text}** {next_char}"
        return match.group(0)

    content = pattern.sub(_insert_space, content)

    content = re.sub(
        r'\*\*\s+([^*]+)\*\*',
        r'**\1**',
        content
    )

    return content


def main():
    if len(sys.argv) < 2:
        print("사용법: python3 fix_multilingual_bold.py <파일경로>")
        sys.exit(1)

    file_path = sys.argv[1]

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        fixed_content = fix_multilingual_bold(content)

        if content != fixed_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(fixed_content)
            print(f"✓ 수정 완료: {file_path}")
        else:
            print(f"○ 수정 불필요: {file_path}")

    except FileNotFoundError:
        print(f"✗ 파일을 찾을 수 없습니다: {file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"✗ 오류 발생: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()