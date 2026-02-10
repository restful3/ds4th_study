// 퀴즈 데이터 (객관식만 추출)
const QUIZ_DATA = {"section1":{"title":"LangChain이란?","summary":"LangChain은 LLM 기반 애플리케이션을 쉽게 개발할 수 있도록 돕는 프레임워크입니다.","quiz":[{"type":"multiple_choice","question":"LangChain의 핵심 특징이 아닌 것은?","options":["a) 10줄 이하로 Agent 생성 가능","b) 표준화된 모델 인터페이스","c) 웹서버가 반드시 필요","d) LangGraph 기반 아키텍처"],"answer":"c","explanation":"LangChain은 웹서버 없이도 사용 가능합니다. Agent는 독립적으로 실행될 수 있습니다."},{"type":"multiple_choice","question":"LangChain이 표준화된 모델 인터페이스를 제공하는 이유는?","options":["a) 코드 실행 속도를 높이기 위해","b) 벤더 종속성(lock-in)을 방지하기 위해","c) 메모리 사용량을 줄이기 위해","d) 디버깅을 쉽게 하기 위해"],"answer":"b","explanation":"표준화된 인터페이스를 통해 OpenAI, Anthropic, Google 등의 프로바이더를 쉽게 교체할 수 있어 벤더 종속성을 방지할 수 있습니다."}]},"section2":{"title":"LangChain의 철학","summary":"LangChain은 LLM이 외부 데이터와 결합할 때 더 강력하며, 미래의 애플리케이션은 점점 더 Agentic 해질 것이라 믿습니다.","quiz":[{"type":"multiple_choice","question":"LangChain의 핵심 신념이 아닌 것은?","options":["a) LLM은 강력한 새로운 기술이다","b) LLM은 외부 데이터와 결합할 때 더 강력하다","c) 프로토타입은 어렵지만 프로덕션은 쉽다","d) 미래의 애플리케이션은 점점 더 Agentic 해질 것이다"],"answer":"c","explanation":"LangChain은 '프로토타입은 쉽지만, 프로덕션은 여전히 어렵다'고 보고 이를 해결하는 것을 목표로 합니다."}]},"section3":{"title":"AI Agent란?","summary":"Agent는 LLM을 추론 엔진으로 사용하여 자율적으로 도구를 호출하고 작업을 수행하는 시스템입니다.","quiz":[{"type":"multiple_choice","question":"Agent와 단순 LLM 호출의 가장 큰 차이는?","options":["a) Agent는 더 빠르다","b) Agent는 외부 도구를 사용할 수 있다","c) Agent는 메모리를 더 적게 사용한다","d) Agent는 무료로 사용할 수 있다"],"answer":"b","explanation":"Agent는 LLM을 추론 엔진으로 사용하여 외부 도구(API, DB 등)를 호출하고 다단계 추론을 수행할 수 있습니다."},{"type":"multiple_choice","question":"ReAct 패턴에서 'ReAct'는 무엇의 약자인가요?","options":["a) Retrieve + Act","b) Reasoning + Acting","c) Request + Action","d) Response + Activate"],"answer":"b","explanation":"ReAct는 Reasoning(추론) + Acting(행동)의 약자로, Agent가 추론하고 행동하는 패턴을 의미합니다."}]},"section4":{"title":"LangChain의 역사와 LangGraph","summary":"LangChain은 2022년 출시 후 지속적으로 발전하여 2025년 v1.0.0에서 하나의 Agent 추상화로 통합되었습니다.","quiz":[{"type":"multiple_choice","question":"LangChain 1.0의 주요 변경사항은?","options":["a) 여러 개의 Chain과 Agent를 추가","b) 모든 Chains와 Agents를 하나의 추상화로 통합","c) 웹 프레임워크 기능 추가","d) 데이터베이스 기능 내장"],"answer":"b","explanation":"LangChain 1.0은 이전의 여러 개의 Agent와 Chain을 create_agent() 하나로 통합했습니다."},{"type":"multiple_choice","question":"LangChain과 LangGraph의 차이로 올바른 것은?","options":["a) LangChain은 저수준, LangGraph는 고수준","b) LangChain은 고수준, LangGraph는 저수준","c) 두 개는 완전히 독립적이다","d) LangGraph가 LangChain을 대체했다"],"answer":"b","explanation":"LangChain은 고수준(High-level)으로 빠르게 Agent를 구축할 수 있고, LangGraph는 저수준(Low-level)으로 복잡한 커스터마이징이 가능합니다."}]},"section5":{"title":"환경 설정 및 실습","summary":"Python 3.10 이상, LangChain 설치, API 키 설정이 필요합니다.","quiz":[{"type":"multiple_choice","question":"LangChain 사용을 위한 최소 Python 버전은?","options":["a) Python 3.8","b) Python 3.9","c) Python 3.10","d) Python 3.12"],"answer":"c","explanation":"LangChain은 Python 3.10 이상을 요구합니다. 3.11이 권장됩니다."},{"type":"multiple_choice","question":"프로토타입 단계에서 추천하는 LLM 프로바이더는?","options":["a) OpenAI GPT-4o-mini (저렴, 빠름, 안정적)","b) Claude Opus (가장 비싸지만 강력)","c) 로컬 모델 (느리지만 무료)","d) Gemini Pro (중간 성능)"],"answer":"a","explanation":"프로토타입 단계에서는 GPT-4o-mini가 저렴하고 빠르며 안정적이어서 가장 적합합니다."}]}};

// 전역 변수
let allQuestions = [];
let userAnswers = {};

// 초기화
document.addEventListener('DOMContentLoaded', () => {
    loadAllQuestions();
});

// 모든 객관식 문제 로드
function loadAllQuestions() {
    allQuestions = [];
    const sections = ['section1', 'section2', 'section3', 'section4', 'section5'];

    sections.forEach(sectionKey => {
        const section = QUIZ_DATA[sectionKey];
        section.quiz.forEach(quiz => {
            if (quiz.type === 'multiple_choice') {
                allQuestions.push(quiz);
            }
        });
    });
}

// 퀴즈 시작
function startQuiz() {
    userAnswers = {};
    document.getElementById('startScreen').style.display = 'none';
    document.getElementById('quizScreen').style.display = 'block';
    renderAllQuestions();
    window.scrollTo(0, 0);
}

// 모든 문제 렌더링
function renderAllQuestions() {
    const container = document.getElementById('quizContainer');
    container.innerHTML = '';

    allQuestions.forEach((quiz, index) => {
        const questionCard = document.createElement('div');
        questionCard.className = 'question-card';
        questionCard.innerHTML = `
            <div class="question-number">문제 ${index + 1}</div>
            <div class="question-text">${quiz.question}</div>
            <div class="options" id="options-${index}">
                ${quiz.options.map(option => `
                    <div class="option" data-question="${index}" data-answer="${option[0]}" onclick="selectAnswer(${index}, '${option[0]}')">
                        ${option}
                    </div>
                `).join('')}
            </div>
        `;
        container.appendChild(questionCard);
    });
}

// 답변 선택
function selectAnswer(questionIndex, answer) {
    // 이전 선택 제거
    const options = document.querySelectorAll(`[data-question="${questionIndex}"]`);
    options.forEach(opt => opt.classList.remove('selected'));

    // 새 선택 표시
    const selectedOption = document.querySelector(`[data-question="${questionIndex}"][data-answer="${answer}"]`);
    if (selectedOption) {
        selectedOption.classList.add('selected');
    }

    // 답변 저장
    userAnswers[questionIndex] = answer;
}

// 퀴즈 제출
function submitQuiz() {
    // 모든 문제에 답했는지 확인
    if (Object.keys(userAnswers).length < allQuestions.length) {
        const unanswered = allQuestions.length - Object.keys(userAnswers).length;
        if (!confirm(`${unanswered}개의 문제가 미답변 상태입니다. 제출하시겠습니까?`)) {
            return;
        }
    }

    // 채점
    let correctCount = 0;
    allQuestions.forEach((quiz, index) => {
        if (userAnswers[index] === quiz.answer) {
            correctCount++;
        }
    });

    // 결과 화면으로 전환
    showResults(correctCount);
}

// 결과 표시
function showResults(correctCount) {
    document.getElementById('quizScreen').style.display = 'none';
    document.getElementById('resultScreen').style.display = 'block';

    const total = allQuestions.length;
    const percentage = ((correctCount / total) * 100).toFixed(1);

    // 통계 표시
    document.getElementById('totalQuestions').textContent = total;
    document.getElementById('correctAnswers').textContent = correctCount;
    document.getElementById('accuracy').textContent = percentage + '%';

    // 메시지 표시
    const messageEl = document.getElementById('resultMessage');
    if (percentage >= 80) {
        messageEl.innerHTML = `
            <h3>🌟 훌륭합니다!</h3>
            <p>Part 1을 완전히 이해하셨습니다!</p>
            <p>이제 Part 2로 넘어가셔도 좋습니다.</p>
        `;
        messageEl.className = 'result-message excellent';
    } else if (percentage >= 60) {
        messageEl.innerHTML = `
            <h3>👍 잘하셨습니다!</h3>
            <p>대부분의 개념을 이해하셨네요.</p>
            <p>틀린 부분을 복습하시면 더 좋습니다.</p>
        `;
        messageEl.className = 'result-message good';
    } else {
        messageEl.innerHTML = `
            <h3>📚 조금 더 복습이 필요합니다</h3>
            <p>교안을 다시 읽어보시고 예제 코드를 실행해보세요.</p>
            <p>경로: <code>docs/part01_introduction.md</code></p>
        `;
        messageEl.className = 'result-message needs-work';
    }

    // 상세 결과 표시
    renderDetailedResults();
    window.scrollTo(0, 0);
}

// 상세 결과 렌더링
function renderDetailedResults() {
    const container = document.getElementById('detailedResults');
    container.innerHTML = '<h3 style="margin-top: 30px;">📝 문제별 상세 결과</h3>';

    allQuestions.forEach((quiz, index) => {
        const userAnswer = userAnswers[index];
        const isCorrect = userAnswer === quiz.answer;

        const resultCard = document.createElement('div');
        resultCard.className = `result-card ${isCorrect ? 'correct' : 'incorrect'}`;

        resultCard.innerHTML = `
            <div class="result-header">
                <span class="result-number">문제 ${index + 1}</span>
                <span class="result-badge ${isCorrect ? 'badge-correct' : 'badge-incorrect'}">
                    ${isCorrect ? '✓ 정답' : '✗ 오답'}
                </span>
            </div>
            <div class="result-question">${quiz.question}</div>
            <div class="result-answers">
                ${quiz.options.map(option => {
                    const letter = option[0];
                    const isUserAnswer = letter === userAnswer;
                    const isCorrectAnswer = letter === quiz.answer;
                    let className = 'result-option';
                    if (isCorrectAnswer) className += ' correct-answer';
                    if (isUserAnswer && !isCorrect) className += ' wrong-answer';

                    return `<div class="${className}">
                        ${option}
                        ${isCorrectAnswer ? ' <strong>(정답)</strong>' : ''}
                        ${isUserAnswer ? ' <strong>(선택)</strong>' : ''}
                    </div>`;
                }).join('')}
            </div>
            <div class="result-explanation">
                <strong>💡 해설:</strong> ${quiz.explanation}
            </div>
        `;

        container.appendChild(resultCard);
    });
}

// 다시 시작
function restartQuiz() {
    userAnswers = {};
    document.getElementById('resultScreen').style.display = 'none';
    document.getElementById('startScreen').style.display = 'block';
    window.scrollTo(0, 0);
}
