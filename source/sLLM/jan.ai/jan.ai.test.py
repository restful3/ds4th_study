import requests
import json

url= "http://localhost:1337/v1/chat/completions"

headers = {
    "Content-Type": "application/json"
}

payload = {
  "messages": [
    {
      "content": "너는 친절한 어시스턴트야. 다음 질문에 답해줘.",
      "role": "system"
    },
    {
      "content": "한국의 역사를 요약해줘",
      "role": "user"
    }
  ],
  "model": "solar-10.7b-slerp",
  "stream": False,
  "max_tokens": 2048,
  "stop": [
    "hello"
  ],
  "frequency_penalty": 0,
  "presence_penalty": 0,
  "temperature": 0.7,
  "top_p": 0.95
}

response = requests.post(url, headers=headers, data=json.dumps(payload))

result = response.json()
content = result['choices'][0]['message']['content']
print(content)