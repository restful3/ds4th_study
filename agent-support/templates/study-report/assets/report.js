(() => {
  const root = document.documentElement;
  const progress = document.getElementById("readingProgress");
  const toc = document.getElementById("reportToc");
  const tocButton = document.getElementById("tocButton");
  const backdrop = document.getElementById("tocBackdrop");
  const tocLinks = [...document.querySelectorAll(".report-toc nav a")];

  function updateProgress() {
    const scrollable = document.documentElement.scrollHeight - window.innerHeight;
    const ratio = scrollable > 0 ? Math.min(1, window.scrollY / scrollable) : 0;
    progress.style.width = `${ratio * 100}%`;
  }

  function setToc(open) {
    toc.classList.toggle("is-open", open);
    tocButton.setAttribute("aria-expanded", String(open));
    backdrop.hidden = !open;
  }

  function applyTheme(theme) {
    root.className = `theme-${theme}`;
    try { localStorage.setItem("ds4th-report-theme", theme); } catch (_) { /* no-op */ }
  }

  try {
    const saved = localStorage.getItem("ds4th-report-theme");
    if (saved === "dark") applyTheme("dark");
  } catch (_) { /* no-op */ }

  document.getElementById("themeButton").addEventListener("click", () => {
    applyTheme(root.classList.contains("theme-dark") ? "light" : "dark");
  });
  document.getElementById("printButton").addEventListener("click", () => window.print());
  document.getElementById("topButton").addEventListener("click", () => window.scrollTo({ top: 0, behavior: "smooth" }));
  tocButton.addEventListener("click", () => setToc(!toc.classList.contains("is-open")));
  backdrop.addEventListener("click", () => setToc(false));
  tocLinks.forEach((link) => link.addEventListener("click", () => setToc(false)));
  document.addEventListener("keydown", (event) => { if (event.key === "Escape") setToc(false); });
  window.addEventListener("scroll", updateProgress, { passive: true });
  window.addEventListener("resize", updateProgress);

  const sections = tocLinks
    .map((link) => document.querySelector(link.getAttribute("href")))
    .filter(Boolean);
  const observer = new IntersectionObserver((entries) => {
    const visible = entries.filter((entry) => entry.isIntersecting).sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
    if (!visible) return;
    tocLinks.forEach((link) => link.classList.toggle("is-active", link.getAttribute("href") === `#${visible.target.id}`));
  }, { rootMargin: "-15% 0px -65%", threshold: [0, 0.25, 0.6] });
  sections.forEach((section) => observer.observe(section));
  updateProgress();
})();
