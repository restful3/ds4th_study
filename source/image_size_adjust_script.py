# 마크다운 이미지를 HTML로 변환하는 스크립트
# re 모듈은 정규표현식을 사용하기 위해 import
import re

def convert_markdown_to_html(markdown_string):
    """
    마크다운 형식의 이미지 문자열을 HTML <img> 태그로 변환하는 함수
    
    Args:
        markdown_string (str): 마크다운 형식의 이미지 문자열
    
    Returns:
        str: HTML 이미지 태그로 변환된 문자열
    """
    lines = markdown_string.strip().split('\n')
    html_lines = []

    for line in lines:
        # 마크다운 이미지 형식(![ ])으로 시작하는 라인 처리
        if line.startswith('!['):
            # 이미지 경로 추출
            image_name = line.split('(')[1].split(')')[0]
            # HTML img 태그로 변환 (너비 800px 고정)
            html = f'<img src="{image_name}" width="800">'
            html_lines.append(html)

    return '\n'.join(html_lines)

# 테스트용 마크다운 입력 문자열
markdown_input = """
![image.png](attachment:40329ccf-a6a4-4586-b792-79003ae0aa94.png)
![image.png](attachment:0afe2dbb-b411-47ad-a16e-75fb3bb782b9.png)
![image.png](attachment:59bf8ff4-d6e6-4bb5-8aec-7a4677b847e0.png)
![image.png](attachment:89b62238-9c53-4dff-aa13-fc027eb16fc2.png)
![image.png](attachment:affa1662-774f-4c42-a791-5f75fb5b92a0.png)
![image.png](attachment:ede1e433-d7f7-4fe1-9f95-2f7ce8bf5387.png)
![image.png](attachment:c53ab781-673b-46f3-a2dc-af59f9f63155.png)
![image.png](attachment:1b76636c-8420-4637-b6c2-7f5f8bd16dea.png)
![image.png](attachment:7aadbd80-cb2c-462d-8462-10487e1bd052.png)
"""

# 마크다운을 HTML로 변환하고 결과 출력
html_output = convert_markdown_to_html(markdown_input)
print(f'"""\n{html_output}\n"""')