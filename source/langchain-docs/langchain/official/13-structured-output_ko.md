# 구조화된 출력

구조화된 출력은 Agent가 특정의 예측 가능한 형식으로 데이터를 반환할 수 있게 합니다. 자연 언어 응답을 파싱하는 대신, JSON 객체, [Pydantic 모델](https://docs.pydantic.dev/latest/) 또는 애플리케이션이 직접 사용할 수 있는 데이터 클래스 형태의 구조화된 데이터를 얻습니다.

LangChain의 `create_agent`는 구조화된 출력을 자동으로 처리합니다. 사용자가 원하는 구조화된 출력 스키마를 설정하면 모델이 구조화된 데이터를 생성할 때 Agent의 상태의 `'structured_response'` 키에서 캡처되고, 검증되고, 반환됩니다.

```python
def create_agent(
    ...
    response_format: Union[
        ToolStrategy[StructuredResponseT],
        ProviderStrategy[StructuredResponseT],
        type[StructuredResponseT],
        None,
    ]
]
```

---

## 응답 형식

`response_format`을 사용하여 Agent가 구조화된 데이터를 반환하는 방법을 제어하세요:

- `ToolStrategy[StructuredResponseT]`: 도구 호출을 구조화된 출력에 사용
- `ProviderStrategy[StructuredResponseT]`: 공급자 네이티브 구조화된 출력 사용
- `type[StructuredResponseT]`: 스키마 타입 - 모델 기능을 기반으로 최적의 전략 자동 선택
- `None`: 구조화된 출력이 명시적으로 요청되지 않음

스키마 타입이 직접 제공되면 LangChain이 자동으로 선택합니다:

- 모델과 선택된 공급자가 네이티브 구조화된 출력을 지원하는 경우 `ProviderStrategy`(예: OpenAI, Anthropic (Claude) 또는 xAI (Grok)).
- 다른 모든 모델의 경우 `ToolStrategy`.

> `langchain>=1.1`을 사용하는 경우 네이티브 구조화된 출력 기능 지원이 모델의 [프로필 데이터](https://reference.langchain.com/python/langchain/chat_models/#langchain.chat_models.profile)에서 동적으로 읽혀집니다. 데이터를 사용할 수 없는 경우 다른 조건을 사용하거나 수동으로 지정하세요:

```python
custom_profile = {
    "structured_output": True,
    # ...
}
model = init_chat_model("...", profile=custom_profile)
```

> 도구가 지정된 경우 모델이 도구와 구조화된 출력을 동시에 사용할 수 있도록 지원해야 합니다.

구조화된 응답은 Agent의 최종 상태의 `structured_response` 키에서 반환됩니다.

---

## Provider 전략

일부 모델 공급자는 API를 통해 구조화된 출력을 기본적으로 지원합니다(예: OpenAI, xAI (Grok), Gemini, Anthropic (Claude)). 사용 가능한 경우 가장 신뢰할 수 있는 방법입니다.

이 전략을 사용하려면 `ProviderStrategy`를 구성하세요:

```python
class ProviderStrategy(Generic[SchemaT]):
    schema: type[SchemaT]
    strict: bool | None = None
```

> `strict` param은 `langchain>=1.2`를 필요로 합니다.

### schema <small>필수</small>

구조화된 출력 형식을 정의하는 스키마. 지원:

- **Pydantic 모델**: 필드 검증이 있는 `BaseModel` 서브클래스. 검증된 Pydantic 인스턴스를 반환합니다.
- **데이터 클래스**: 타입 주석이 있는 Python 데이터 클래스. dict를 반환합니다.
- **TypedDict**: 타입 지정된 딕셔너리 클래스. dict를 반환합니다.
- **JSON 스키마**: JSON 스키마 사양이 있는 딕셔너리. dict를 반환합니다.

### strict

선택적 부울 매개변수로 엄격한 스키마 준수를 활성화합니다. 일부 공급자(예: OpenAI 및 xAI)에서 지원됩니다. 기본값은 `None`(비활성화)입니다.

LangChain은 스키마 타입을 `create_agent.response_format`에 직접 전달하고 모델이 네이티브 구조화된 출력을 지원할 때 자동으로 `ProviderStrategy`를 사용합니다:

#### Pydantic 모델

```python
from pydantic import BaseModel, Field
from langchain.agents import create_agent

class ContactInfo(BaseModel):
    """사람의 연락처 정보입니다."""
    name: str = Field(description="사람의 이름")
    email: str = Field(description="사람의 이메일 주소")
    phone: str = Field(description="사람의 전화 번호")

agent = create_agent(
    model="gpt-5",
    response_format=ContactInfo  # 자동으로 ProviderStrategy 선택
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Extract contact info from: John Doe, john@example.com, (555) 123-4567"}]
})

print(result["structured_response"])
# ContactInfo(name='John Doe', email='john@example.com', phone='(555) 123-4567')
```

#### 데이터 클래스

```python
from dataclasses import dataclass
from langchain.agents import create_agent

@dataclass
class ContactInfo:
    """사람의 연락처 정보입니다."""
    name: str  # 사람의 이름
    email: str  # 사람의 이메일 주소
    phone: str  # 사람의 전화 번호

agent = create_agent(
    model="gpt-5",
    tools=tools,
    response_format=ContactInfo  # 자동으로 ProviderStrategy 선택
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Extract contact info from: John Doe, john@example.com, (555) 123-4567"}]
})

result["structured_response"]
# {'name': 'John Doe', 'email': 'john@example.com', 'phone': '(555) 123-4567'}
```

#### TypedDict

```python
from typing_extensions import TypedDict
from langchain.agents import create_agent

class ContactInfo(TypedDict):
    """사람의 연락처 정보입니다."""
    name: str  # 사람의 이름
    email: str  # 사람의 이메일 주소
    phone: str  # 사람의 전화 번호

agent = create_agent(
    model="gpt-5",
    tools=tools,
    response_format=ContactInfo  # 자동으로 ProviderStrategy 선택
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Extract contact info from: John Doe, john@example.com, (555) 123-4567"}]
})

result["structured_response"]
# {'name': 'John Doe', 'email': 'john@example.com', 'phone': '(555) 123-4567'}
```

#### JSON 스키마

```python
from langchain.agents import create_agent

contact_info_schema = {
    "type": "object",
    "description": "사람의 연락처 정보입니다.",
    "properties": {
        "name": {"type": "string", "description": "사람의 이름"},
        "email": {"type": "string", "description": "사람의 이메일 주소"},
        "phone": {"type": "string", "description": "사람의 전화 번호"}
    },
    "required": ["name", "email", "phone"]
}

agent = create_agent(
    model="gpt-5",
    tools=tools,
    response_format=ProviderStrategy(contact_info_schema)
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Extract contact info from: John Doe, john@example.com, (555) 123-4567"}]
})

result["structured_response"]
# {'name': 'John Doe', 'email': 'john@example.com', 'phone': '(555) 123-4567'}
```

공급자 네이티브 구조화된 출력은 높은 신뢰성과 엄격한 검증을 제공합니다. 모델 공급자가 스키마를 적용하기 때문입니다. 사용 가능할 때 이를 사용하세요.

> 공급자가 모델 선택에 대해 구조화된 출력을 기본적으로 지원하는 경우 `response_format=ProductReview` 대신 `response_format=ProviderStrategy(ProductReview)`를 작성하는 것과 기능적으로 동일합니다.
>
> 어느 경우든 구조화된 출력이 지원되지 않으면 Agent가 도구 호출 전략으로 폴백됩니다.

---

## 도구 호출 전략

네이티브 구조화된 출력을 지원하지 않는 모델의 경우 LangChain은 도구 호출을 사용하여 동일한 결과를 달성합니다. 이는 도구 호출을 지원하는 모든 모델(대부분의 최신 모델)과 함께 작동합니다.

이 전략을 사용하려면 `ToolStrategy`를 구성하세요:

```python
class ToolStrategy(Generic[SchemaT]):
    schema: type[SchemaT]
    tool_message_content: str | None
    handle_errors: Union[
        bool,
        str,
        type[Exception],
        tuple[type[Exception], ...],
        Callable[[Exception], str],
    ]
```

### schema <small>필수</small>

구조화된 출력 형식을 정의하는 스키마. 지원:

- **Pydantic 모델**: 필드 검증이 있는 `BaseModel` 서브클래스. 검증된 Pydantic 인스턴스를 반환합니다.
- **데이터 클래스**: 타입 주석이 있는 Python 데이터 클래스. dict를 반환합니다.
- **TypedDict**: 타입 지정된 딕셔너리 클래스. dict를 반환합니다.
- **JSON 스키마**: JSON 스키마 사양이 있는 딕셔너리. dict를 반환합니다.
- **Union 타입**: 여러 스키마 옵션. 모델이 컨텍스트를 기반으로 가장 적절한 스키마를 선택합니다.

### tool_message_content

구조화된 출력이 생성될 때 반환되는 도구 메시지에 대한 사용자 정의 콘텐츠. 제공되지 않으면 구조화된 응답 데이터를 표시하는 메시지로 기본값이 지정됩니다.

### handle_errors

구조화된 출력 검증 실패에 대한 오류 처리 전략. 기본값은 `True`입니다.

- `True`: 기본 오류 템플릿으로 모든 오류 포착
- `str`: 이 사용자 정의 메시지로 모든 오류 포착
- `type[Exception]`: 기본 메시지로만 이 예외 타입 포착
- `tuple[type[Exception], ...]`: 기본 메시지로만 이러한 예외 타입 포착
- `Callable[[Exception], str]`: 오류 메시지를 반환하는 사용자 정의 함수
- `False`: 재시도 없음, 예외 전파 허용

#### Pydantic 모델

```python
from pydantic import BaseModel, Field
from typing import Literal
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy

class ProductReview(BaseModel):
    """제품 리뷰 분석입니다."""
    rating: int | None = Field(description="제품의 평가", ge=1, le=5)
    sentiment: Literal["positive", "negative"] = Field(description="리뷰의 감정")
    key_points: list[str] = Field(description="리뷰의 핵심 포인트. 소문자, 각 1-3단어.")

agent = create_agent(
    model="gpt-5",
    tools=tools,
    response_format=ToolStrategy(ProductReview)
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Analyze this review: 'Great product: 5 out of 5 stars. Fast shipping, but expensive'"}]
})

result["structured_response"]
# ProductReview(rating=5, sentiment='positive', key_points=['fast shipping', 'expensive'])
```

#### 데이터 클래스

```python
from dataclasses import dataclass
from typing import Literal
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy

@dataclass
class ProductReview:
    """제품 리뷰 분석입니다."""
    rating: int | None  # 제품의 평가 (1-5)
    sentiment: Literal["positive", "negative"]  # 리뷰의 감정
    key_points: list[str]  # 리뷰의 핵심 포인트

agent = create_agent(
    model="gpt-5",
    tools=tools,
    response_format=ToolStrategy(ProductReview)
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Analyze this review: 'Great product: 5 out of 5 stars. Fast shipping, but expensive'"}]
})

result["structured_response"]
# {'rating': 5, 'sentiment': 'positive', 'key_points': ['fast shipping', 'expensive']}
```

#### TypedDict

```python
from typing_extensions import TypedDict
from typing import Literal
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy

class ProductReview(TypedDict):
    """제품 리뷰 분석입니다."""
    rating: int | None  # 제품의 평가 (1-5)
    sentiment: Literal["positive", "negative"]  # 리뷰의 감정
    key_points: list[str]  # 리뷰의 핵심 포인트

agent = create_agent(
    model="gpt-5",
    tools=tools,
    response_format=ToolStrategy(ProductReview)
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Analyze this review: 'Great product: 5 out of 5 stars. Fast shipping, but expensive'"}]
})

result["structured_response"]
# {'rating': 5, 'sentiment': 'positive', 'key_points': ['fast shipping', 'expensive']}
```

#### JSON 스키마

```python
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy

product_review_schema = {
    "type": "object",
    "description": "제품 리뷰 분석입니다.",
    "properties": {
        "rating": {
            "type": ["integer", "null"],
            "description": "제품의 평가 (1-5)",
            "minimum": 1,
            "maximum": 5
        },
        "sentiment": {
            "type": "string",
            "enum": ["positive", "negative"],
            "description": "리뷰의 감정"
        },
        "key_points": {
            "type": "array",
            "items": {"type": "string"},
            "description": "리뷰의 핵심 포인트"
        }
    },
    "required": ["sentiment", "key_points"]
}

agent = create_agent(
    model="gpt-5",
    tools=tools,
    response_format=ToolStrategy(product_review_schema)
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Analyze this review: 'Great product: 5 out of 5 stars. Fast shipping, but expensive'"}]
})

result["structured_response"]
# {'rating': 5, 'sentiment': 'positive', 'key_points': ['fast shipping', 'expensive']}
```

#### Union 타입

```python
from pydantic import BaseModel, Field
from typing import Literal, Union
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy

class ProductReview(BaseModel):
    """제품 리뷰 분석입니다."""
    rating: int | None = Field(description="제품의 평가", ge=1, le=5)
    sentiment: Literal["positive", "negative"] = Field(description="리뷰의 감정")
    key_points: list[str] = Field(description="리뷰의 핵심 포인트. 소문자, 각 1-3단어.")

class CustomerComplaint(BaseModel):
    """제품 또는 서비스에 대한 고객 불만입니다."""
    issue_type: Literal["product", "service", "shipping", "billing"] = Field(description="문제의 유형")
    description: str = Field(description="불만의 설명")
    urgency: Literal["low", "medium", "high"] = Field(description="긴급도")

agent = create_agent(
    model="gpt-5",
    tools=[],
    response_format=ToolStrategy(Union[ProductReview, CustomerComplaint])  # 기본값: handle_errors=True
)

agent.invoke({
    "messages": [{"role": "user", "content": "Analyze this review: 'Great product! 5 stars'"}]
})
```

---

## 사용자 정의 도구 메시지 콘텐츠

`tool_message_content` 매개변수는 구조화된 출력이 생성될 때 대화 이력에 나타나는 메시지를 사용자 정의할 수 있습니다:

```python
from pydantic import BaseModel, Field
from typing import Literal
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy

class MeetingAction(BaseModel):
    """회의 기록에서 추출한 액션 항목입니다."""
    task: str = Field(description="완료할 특정 작업")
    assignee: str = Field(description="작업을 담당할 사람")
    priority: Literal["low", "medium", "high"] = Field(description="우선순위 수준")

agent = create_agent(
    model="gpt-5",
    tools=[],
    response_format=ToolStrategy(
        schema=MeetingAction,
        tool_message_content="Action item captured and added to meeting notes!"
    )
)

agent.invoke({
    "messages": [{"role": "user", "content": "From our meeting: Sarah needs to update the project timeline as soon as possible"}]
})
```

```text
================================ Human Message =================================

From our meeting: Sarah needs to update the project timeline as soon as possible
================================== Ai Message ==================================
Tool Calls:
  MeetingAction (call_1)
 Call ID: call_1
  Args:
    task: Update the project timeline
    assignee: Sarah
    priority: high
================================= Tool Message =================================
Name: MeetingAction

Action item captured and added to meeting notes!
```

`tool_message_content`를 사용하지 않으면 최종 ToolMessage는:

```text
================================= Tool Message =================================
Name: MeetingAction

Returning structured response: {'task': 'update the project timeline', 'assignee': 'Sarah', 'priority': 'high'}
```

---

## 오류 처리

모델은 도구 호출을 통해 구조화된 출력을 생성할 때 실수할 수 있습니다. LangChain은 이러한 오류를 자동으로 처리하는 지능형 재시도 메커니즘을 제공합니다.

### 여러 구조화된 출력 오류

모델이 잘못하여 여러 구조화된 출력 도구를 호출할 때 Agent는 `ToolMessage`에 오류 피드백을 제공하고 모델이 재시도하도록 프롬프트합니다:

```python
from pydantic import BaseModel, Field
from typing import Union
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy

class ContactInfo(BaseModel):
    name: str = Field(description="사람의 이름")
    email: str = Field(description="이메일 주소")

class EventDetails(BaseModel):
    event_name: str = Field(description="이벤트의 이름")
    date: str = Field(description="이벤트 날짜")

agent = create_agent(
    model="gpt-5",
    tools=[],
    response_format=ToolStrategy(Union[ContactInfo, EventDetails])  # 기본값: handle_errors=True
)

agent.invoke({
    "messages": [{"role": "user", "content": "Extract info: John Doe (john@email.com) is organizing Tech Conference on March 15th"}]
})
```

```text
================================ Human Message =================================

Extract info: John Doe (john@email.com) is organizing Tech Conference on March 15th
None
================================== Ai Message ==================================
Tool Calls:
  ContactInfo (call_1)
 Call ID: call_1
  Args:
    name: John Doe
    email: john@email.com
  EventDetails (call_2)
 Call ID: call_2
  Args:
    event_name: Tech Conference
    date: March 15th
================================= Tool Message =================================
Name: ContactInfo

Error: Model incorrectly returned multiple structured responses (ContactInfo, EventDetails) when only one is expected. Please fix your mistakes.
================================= Tool Message =================================
Name: EventDetails

Error: Model incorrectly returned multiple structured responses (ContactInfo, EventDetails) when only one is expected. Please fix your mistakes.
================================== Ai Message ==================================
Tool Calls:
  ContactInfo (call_3)
 Call ID: call_3
  Args:
    name: John Doe
    email: john@email.com
================================= Tool Message =================================
Name: ContactInfo

Returning structured response: {'name': 'John Doe', 'email': 'john@email.com'}
```

### 스키마 검증 오류

구조화된 출력이 예상 스키마와 일치하지 않을 때 Agent는 특정 오류 피드백을 제공합니다:

```python
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy

class ProductRating(BaseModel):
    rating: int | None = Field(description="1-5 범위의 평가", ge=1, le=5)
    comment: str = Field(description="리뷰 코멘트")

agent = create_agent(
    model="gpt-5",
    tools=[],
    response_format=ToolStrategy(ProductRating),  # 기본값: handle_errors=True
    system_prompt="You are a helpful assistant that parses product reviews. Do not make any field or value up."
)

agent.invoke({
    "messages": [{"role": "user", "content": "Parse this: Amazing product, 10/10!"}]
})
```

```text
================================ Human Message =================================

Parse this: Amazing product, 10/10!
================================== Ai Message ==================================
Tool Calls:
  ProductRating (call_1)
 Call ID: call_1
  Args:
    rating: 10
    comment: Amazing product
================================= Tool Message =================================
Name: ProductRating

Error: Failed to parse structured output for tool 'ProductRating': 1 validation error for ProductRating.rating
  Input should be less than or equal to 5 [type=less_than_equal, input_value=10, input_type=int]. Please fix your mistakes.
================================== Ai Message ==================================
Tool Calls:
  ProductRating (call_2)
 Call ID: call_2
  Args:
    rating: 5
    comment: Amazing product
================================= Tool Message =================================
Name: ProductRating

Returning structured response: {'rating': 5, 'comment': 'Amazing product'}
```

---

## 오류 처리 전략

`handle_errors` 매개변수를 사용하여 오류 처리 방식을 사용자 정의할 수 있습니다:

**사용자 정의 오류 메시지:**

```python
ToolStrategy(
    schema=ProductRating,
    handle_errors="Please provide a valid rating between 1-5 and include a comment."
)
```

`handle_errors`가 문자열인 경우 Agent는 고정된 도구 메시지로 모델이 다시 시도하도록 항상 프롬프트합니다:

```text
================================= Tool Message =================================
Name: ProductRating

Please provide a valid rating between 1-5 and include a comment.
```

**특정 예외만 처리:**

```python
ToolStrategy(
    schema=ProductRating,
    handle_errors=ValueError  # ValueError만 재시도하고 다른 것은 발생
)
```

`handle_errors`가 예외 타입인 경우 Agent는 발생한 예외가 지정된 타입인 경우에만 재시도합니다(기본 오류 메시지 사용). 다른 모든 경우에는 예외가 발생합니다.

**여러 예외 타입 처리:**

```python
ToolStrategy(
    schema=ProductRating,
    handle_errors=(ValueError, TypeError)  # ValueError 및 TypeError에서 재시도
)
```

`handle_errors`가 예외 튜플인 경우 Agent는 발생한 예외가 지정된 타입 중 하나인 경우에만 재시도합니다(기본 오류 메시지 사용). 다른 모든 경우에는 예외가 발생합니다.

**사용자 정의 오류 처리 함수:**

```python
from langchain.agents.structured_output import StructuredOutputValidationError
from langchain.agents.structured_output import MultipleStructuredOutputsError

def custom_error_handler(error: Exception) -> str:
    if isinstance(error, StructuredOutputValidationError):
        return "There was an issue with the format. Try again."
    elif isinstance(error, MultipleStructuredOutputsError):
        return "Multiple structured outputs were returned. Pick the most relevant one."
    else:
        return f"Error: {str(error)}"

agent = create_agent(
    model="gpt-5",
    tools=[],
    response_format=ToolStrategy(
        schema=Union[ContactInfo, EventDetails],
        handle_errors=custom_error_handler
    )  # 기본값: handle_errors=True
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Extract info: John Doe (john@email.com) is organizing Tech Conference on March 15th"}]
})

for msg in result['messages']:
    # 메시지가 실제로 ToolMessage 객체인 경우(dict가 아님) 클래스 이름을 확인합니다
    if type(msg).__name__ == "ToolMessage":
        print(msg.content)
    # 메시지가 딕셔너리이거나 폴백이 필요한 경우
    elif isinstance(msg, dict) and msg.get('tool_call_id'):
        print(msg['content'])
```

`StructuredOutputValidationError`에 대해:

```text
================================= Tool Message =================================
Name: ToolStrategy

There was an issue with the format. Try again.
```

`MultipleStructuredOutputsError`에 대해:

```text
================================= Tool Message =================================
Name: ToolStrategy

Multiple structured outputs were returned. Pick the most relevant one.
```

다른 오류에 대해:

```text
================================= Tool Message =================================
Name: ToolStrategy

Error: <error message>
```

**오류 처리 없음:**

```python
response_format = ToolStrategy(
    schema=ProductRating,
    handle_errors=False  # 모든 오류 발생
)
```
