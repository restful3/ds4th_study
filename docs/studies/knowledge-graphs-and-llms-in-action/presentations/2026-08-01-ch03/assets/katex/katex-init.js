// 자체 호스팅 KaTeX 로 인라인/디스플레이 수식 렌더 (오프라인, 외부 CDN 없음).
// 코드/코드블록의 '$' 는 수식으로 오인식하지 않도록 제외한다.
document.addEventListener("DOMContentLoaded", function () {
  if (typeof renderMathInElement !== "function") return;
  renderMathInElement(document.body, {
    delimiters: [
      { left: "$$", right: "$$", display: true },
      { left: "\\[", right: "\\]", display: true },
      { left: "\\(", right: "\\)", display: false },
      { left: "$", right: "$", display: false }
    ],
    throwOnError: false,
    ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code", "option"]
  });
});
