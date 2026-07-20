// AI Odyssey Publisher — Report 런타임 (인터랙티브 HTML + 인쇄 프리프로세스)
// 슬라이드 `deck.js` 에서 네비게이션·뷰포트 스케일링을 제거하고,
// 리포트에 필요한 (1) 테마 토글 (2) Chart.js 빌더
// (4) IntersectionObserver 기반 카운터/애니메이션 (5) TOC 자동 생성
// (6) 각주 번호 자동 부여만 남겼다.

(function () {
  // ===== Theme toggle =====
  var THEME_KEY = 'aio-report-theme';
  var saved = localStorage.getItem(THEME_KEY);
  var current = saved || (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');

  window.applyTheme = function (t) {
    document.documentElement.className = 'theme-' + t;
    var btn = document.getElementById('themeToggle');
    if (btn) {
      btn.textContent = t === 'dark' ? '☾' : '☀';
      btn.setAttribute('aria-pressed', t === 'dark' ? 'true' : 'false');
      btn.setAttribute('title', t === 'dark' ? '라이트 모드로 전환' : '나이트 모드로 전환');
    }
    localStorage.setItem(THEME_KEY, t);
    current = t;
    if (typeof onThemeChange === 'function') onThemeChange();
  };
  window.cycleTheme = function () { window.applyTheme(current === 'dark' ? 'light' : 'dark'); };
  window.applyTheme(current);

  // ===== Floating controls / drawer TOC =====
  function getHeadingLabel(h) {
    return (h.textContent || '').replace(/\s+/g, ' ').trim();
  }

  function ensureHeadingId(h, idx) {
    if (h.id) return h.id;
    h.id = 'toc-heading-' + idx;
    return h.id;
  }

  window.buildDrawerTOC = function () {
    var list = document.getElementById('tocDrawerList');
    if (!list || list.children.length) return;
    var headings = Array.from(document.querySelectorAll('.report-section h1, .report-section h2, .report-appendix h1, .report-appendix h2'));
    headings.forEach(function (h, idx) {
      var label = getHeadingLabel(h);
      if (!label || /^References$/i.test(label)) label = '참고문헌';
      var id = ensureHeadingId(h, idx + 1);
      var li = document.createElement('li');
      li.className = 'report-toc-drawer__item report-toc-drawer__item--' + h.tagName.toLowerCase();
      var a = document.createElement('a');
      a.href = '#' + id;
      a.textContent = label;
      a.addEventListener('click', function () { window.closeTOCDrawer(); });
      li.appendChild(a);
      list.appendChild(li);
    });
  };

  window.openTOCDrawer = function () {
    window.buildDrawerTOC();
    var drawer = document.getElementById('tocDrawer');
    var backdrop = document.getElementById('tocBackdrop');
    var toggle = document.getElementById('tocToggle');
    if (!drawer || !backdrop) return;
    drawer.classList.add('is-open');
    drawer.setAttribute('aria-hidden', 'false');
    backdrop.hidden = false;
    backdrop.classList.add('is-open');
    if (toggle) toggle.setAttribute('aria-expanded', 'true');
  };

  window.closeTOCDrawer = function () {
    var drawer = document.getElementById('tocDrawer');
    var backdrop = document.getElementById('tocBackdrop');
    var toggle = document.getElementById('tocToggle');
    if (!drawer || !backdrop) return;
    drawer.classList.remove('is-open');
    drawer.setAttribute('aria-hidden', 'true');
    backdrop.classList.remove('is-open');
    backdrop.hidden = true;
    if (toggle) toggle.setAttribute('aria-expanded', 'false');
  };

  window.openSettingsPanel = function () {
    var panel = document.getElementById('settingsPanel');
    var toggle = document.getElementById('settingsToggle');
    if (!panel || !toggle) return;
    panel.classList.add('is-open');
    panel.setAttribute('aria-hidden', 'false');
    toggle.setAttribute('aria-expanded', 'true');
  };

  window.closeSettingsPanel = function () {
    var panel = document.getElementById('settingsPanel');
    var toggle = document.getElementById('settingsToggle');
    if (!panel || !toggle) return;
    panel.classList.remove('is-open');
    panel.setAttribute('aria-hidden', 'true');
    toggle.setAttribute('aria-expanded', 'false');
  };

  window.toggleSettingsPanel = function () {
    var panel = document.getElementById('settingsPanel');
    if (panel && panel.classList.contains('is-open')) window.closeSettingsPanel();
    else window.openSettingsPanel();
  };

  function setupFloatingControls() {
    var tocToggle = document.getElementById('tocToggle');
    var tocClose = document.getElementById('tocClose');
    var backdrop = document.getElementById('tocBackdrop');
    var settingsToggle = document.getElementById('settingsToggle');
    if (tocToggle) tocToggle.addEventListener('click', window.openTOCDrawer);
    if (tocClose) tocClose.addEventListener('click', window.closeTOCDrawer);
    if (backdrop) backdrop.addEventListener('click', window.closeTOCDrawer);
    if (settingsToggle) settingsToggle.addEventListener('click', window.toggleSettingsPanel);
    document.addEventListener('keydown', function (e) {
      if (e.key === 'Escape') {
        window.closeTOCDrawer();
        window.closeSettingsPanel();
      }
    });
    document.addEventListener('click', function (e) {
      var settings = document.querySelector('.report-settings');
      if (settings && !settings.contains(e.target)) window.closeSettingsPanel();
    });

    // Auto-hide floating controls — Medium/Substack scroll-direction reveal +
    // edge-hover boost. idle timer 가 아니라 사용자 의도(읽기 vs 탐색)에 연동.
    // - 아래로 스크롤 = 읽는 중 → 숨김
    // - 위로 스크롤 = 탐색 의도 → 표시
    // - scrollY < 100 (페이지 최상단 부근) = 항상 표시
    // - 마우스가 좌/우 가장자리 100px 영역 = 즉시 표시
    // - TOC 드로어 / 설정 패널 열림 = 강제 표시
    var SCROLL_THRESHOLD = 8;
    var TOP_ZONE = 100;
    var EDGE_ZONE = 100;
    var lastScrollY = window.scrollY || window.pageYOffset || 0;

    function isAnyPanelOpen() {
      var drawer = document.getElementById('tocDrawer');
      var panel = document.getElementById('settingsPanel');
      return (drawer && drawer.classList.contains('is-open'))
          || (panel && panel.classList.contains('is-open'));
    }
    function showChrome() {
      if (!document.body.classList.contains('chrome-hidden')) return;
      document.body.classList.remove('chrome-hidden');
    }
    function hideChrome() {
      if (isAnyPanelOpen()) return;
      if ((window.scrollY || 0) < TOP_ZONE) return;
      if (document.body.classList.contains('chrome-hidden')) return;
      document.body.classList.add('chrome-hidden');
    }
    function onScroll() {
      var y = window.scrollY || window.pageYOffset || 0;
      var dy = y - lastScrollY;
      if (Math.abs(dy) < SCROLL_THRESHOLD) return;
      if (y < TOP_ZONE) showChrome();
      else if (dy > 0) hideChrome();
      else showChrome();
      lastScrollY = y;
    }
    function onMouseMove(e) {
      if (e.clientX <= EDGE_ZONE || e.clientX >= window.innerWidth - EDGE_ZONE) {
        showChrome();
      }
    }
    document.addEventListener('scroll', onScroll, { passive: true });
    document.addEventListener('mousemove', onMouseMove, { passive: true });
    document.addEventListener('touchstart', showChrome, { passive: true });
    document.addEventListener('keydown', showChrome);
  }

  // ===== TOC auto-generation =====
  window.buildTOC = function () {
    var tocList = document.querySelector('.report-toc__list');
    if (!tocList) return;
    // 이미 수동으로 채워져 있으면 덮어쓰지 않음
    if (tocList.children.length > 0) return;

    function makeItem(href, num, label, isSub) {
      var li = document.createElement('li');
      li.className = 'report-toc__item' + (isSub ? ' report-toc__item--h3' : '');
      li.innerHTML =
        '<a href="#' + href + '">' +
          '<span class="report-toc__num">' + num + '</span>' +
          '<span class="report-toc__label">' + label + '</span>' +
          '<span class="report-toc__leader"></span>' +
          '<span class="report-toc__page" data-toc-page></span>' +
        '</a>';
      return li;
    }

    function isAppendixSection(sec) {
      if (sec.classList.contains('report-appendix')) return true;
      var k = sec.querySelector(':scope > .report-section__kicker');
      return !!(k && /^Appendix/i.test(k.textContent));
    }

    var secCount = 0;
    var subCount = 0;

    // 1) 일반 섹션 — 부록(Appendix) 이 아닌 .report-section
    document.querySelectorAll('.report-section').forEach(function (sec) {
      if (isAppendixSection(sec)) return;
      var heading = sec.querySelector(':scope > h1, :scope > h2');
      if (heading) {
        secCount += 1;
        subCount = 0;
        if (!sec.id) sec.id = 'section-' + secCount;
        tocList.appendChild(makeItem(sec.id, String(secCount).padStart(2, '0'), heading.textContent, false));
      }
      sec.querySelectorAll(':scope h3').forEach(function (h3) {
        subCount += 1;
        if (!h3.id) h3.id = (sec.id || 'section-' + secCount) + '-' + subCount;
        tocList.appendChild(makeItem(h3.id, secCount + '.' + subCount, h3.textContent, true));
      });
    });

    // 2) 참고문헌 — report 말미의 독립 섹션
    var refs = document.querySelector('.report-appendix.s-refs');
    if (refs) {
      var refsHeading = refs.querySelector(':scope > h1, :scope > h2');
      secCount += 1;
      if (!refs.id) refs.id = 'section-' + secCount;
      if (refsHeading && !refsHeading.id) refsHeading.id = refs.id + '-' + refsHeading.tagName.toLowerCase();
      tocList.appendChild(makeItem(refs.id, String(secCount).padStart(2, '0'), refsHeading ? refsHeading.textContent : '참고문헌', false));
    }
  };

  // ===== Footnote auto-numbering =====
  window.numberFootnotes = function () {
    var refs = document.querySelectorAll('.footnote-ref');
    refs.forEach(function (ref, idx) {
      var n = idx + 1;
      if (!ref.textContent.trim()) ref.textContent = n;
      if (!ref.id) ref.id = 'fnref-' + n;
      if (!ref.getAttribute('href')) ref.setAttribute('href', '#fn-' + n);
    });
    var items = document.querySelectorAll('.footnotes li');
    items.forEach(function (li, idx) {
      var n = idx + 1;
      if (!li.id) li.id = 'fn-' + n;
    });
  };

  // ===== Counter animation (data-count, IntersectionObserver) =====
  function animateCounter(el) {
    if (el.dataset.counted === '1') return;
    el.dataset.counted = '1';
    var target = parseFloat(el.dataset.count);
    var suffix = el.dataset.suffix || '';
    var prefix = el.dataset.prefix || '';
    var isFloat = String(target).indexOf('.') > -1;
    var start = performance.now();
    var duration = 1100;
    (function tick(now) {
      var p = Math.min((now - start) / duration, 1);
      var eased = 1 - Math.pow(1 - p, 3);
      var v = target * eased;
      el.textContent = prefix + (isFloat ? v.toFixed(1) : Math.round(v).toLocaleString()) + suffix;
      if (p < 1) requestAnimationFrame(tick);
    })(start);
  }
  function setupObserver() {
    if (!('IntersectionObserver' in window)) {
      // Fallback: 즉시 최종 값으로 채움 (인쇄 경로 안전장치)
      document.querySelectorAll('[data-count]').forEach(function (el) {
        var target = parseFloat(el.dataset.count);
        var isFloat = String(target).indexOf('.') > -1;
        el.textContent = (el.dataset.prefix || '') + (isFloat ? target.toFixed(1) : Math.round(target).toLocaleString()) + (el.dataset.suffix || '');
      });
      document.querySelectorAll('.animate-in').forEach(function (el) { el.classList.add('is-visible'); });
      return;
    }
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (!e.isIntersecting) return;
        var el = e.target;
        if (el.hasAttribute('data-count')) animateCounter(el);
        if (el.classList.contains('animate-in')) el.classList.add('is-visible');
        io.unobserve(el);
      });
    }, { threshold: 0.25 });
    document.querySelectorAll('[data-count]').forEach(function (el) { io.observe(el); });
    document.querySelectorAll('.animate-in').forEach(function (el) { io.observe(el); });
  }

  // ===== Chart.js (슬라이드 bars 빌더 + 리포트 전용 positioningMap scatter) =====
  var chartsBuilt = false;
  var barsChart = null;
  var positioningChart = null;
  function getChartColors() {
    var s = getComputedStyle(document.documentElement);
    return {
      text: s.getPropertyValue('--text').trim(),
      textSecondary: s.getPropertyValue('--text-secondary').trim(),
      border: s.getPropertyValue('--border').trim(),
      surface: s.getPropertyValue('--surface').trim(),
      lgRed: s.getPropertyValue('--brand-primary').trim(),
      lgRedDeep: s.getPropertyValue('--brand-deep').trim(),
    };
  }
  function resetCanvas(id) {
    var old = document.getElementById(id);
    if (!old) return null;
    var parent = old.parentNode;
    var c = document.createElement('canvas');
    c.id = id;
    c.setAttribute('role', 'img');
    c.setAttribute('aria-label', 'Bars chart');
    parent.replaceChild(c, old);
    return c;
  }
  function buildPositioningMap(c) {
    try { if (positioningChart) { positioningChart.destroy(); positioningChart = null; } } catch (e) {}
    var canvas = resetCanvas('positioningMap');
    if (!canvas) return;
    // x: 0 = 개발자/API, 100 = 비개발자/UI · y: 0 = 좁음(브라우저), 100 = 넓음(OS 전체)
    var points = [
      { label: 'Anthropic',       x: 18, y: 92, color: '#0F2C59' },
      { label: 'OpenAI Operator', x: 82, y: 35, color: '#10b981' },
      { label: 'Google AI Ultra', x: 55, y: 58, color: '#2563eb' },
    ];
    var datasets = points.map(function (p) {
      return {
        label: p.label,
        data: [{ x: p.x, y: p.y, _label: p.label }],
        backgroundColor: p.color,
        borderColor: p.color,
        pointRadius: 11,
        pointHoverRadius: 13,
        pointStyle: 'circle',
      };
    });
    positioningChart = new Chart(canvas.getContext('2d'), {
      type: 'scatter',
      data: { datasets: datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        layout: { padding: { top: 18, right: 18, bottom: 6, left: 6 } },
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: c.text, titleColor: c.surface, bodyColor: c.surface,
            padding: 8, cornerRadius: 6, displayColors: false,
            callbacks: {
              title: function (items) { return items[0].dataset.label; },
              label: function (ctx) {
                var xLbl = ctx.parsed.x >= 50 ? '비개발자/UI' : '개발자/API';
                var yLbl = ctx.parsed.y >= 50 ? '넓은 범위' : '좁은 범위';
                return xLbl + ' · ' + yLbl;
              }
            }
          }
        },
        scales: {
          x: {
            min: 0, max: 100,
            ticks: {
              color: c.textSecondary, font: { size: 9, family: 'Inter', weight: '600' },
              callback: function (v) {
                if (v === 0) return '개발자/API';
                if (v === 100) return '비개발자/UI';
                if (v === 50) return '혼합';
                return '';
              }
            },
            grid: { color: c.border, drawBorder: false },
            border: { display: false },
            title: { display: false }
          },
          y: {
            min: 0, max: 100,
            ticks: {
              color: c.textSecondary, font: { size: 9, family: 'Inter', weight: '600' },
              callback: function (v) {
                if (v === 0) return '좁음(브라우저)';
                if (v === 100) return '넓음(OS 전체)';
                if (v === 50) return '중간';
                return '';
              }
            },
            grid: { color: c.border, drawBorder: false },
            border: { display: false }
          }
        }
      },
      plugins: [{
        id: 'pointLabels',
        afterDatasetsDraw: function (chart) {
          var ctx = chart.ctx;
          ctx.save();
          ctx.font = "700 10.5px 'Inter', 'Noto Sans KR', sans-serif";
          ctx.fillStyle = c.text;
          ctx.textAlign = 'left';
          ctx.textBaseline = 'middle';
          chart.data.datasets.forEach(function (ds, i) {
            var meta = chart.getDatasetMeta(i);
            meta.data.forEach(function (pt) {
              ctx.fillText('  ' + ds.label, pt.x + 10, pt.y);
            });
          });
          ctx.restore();
        }
      }]
    });
  }

  window.buildCharts = function () {
    if (typeof Chart === 'undefined') return;
    if (chartsBuilt) return;
    var c = getChartColors();
    // 포지셔닝 맵이 본문에 있으면 그린다 (없으면 skip)
    buildPositioningMap(c);
    try { if (barsChart) { barsChart.destroy(); barsChart = null; } } catch (e) {}
    var ctx = resetCanvas('barsChart');
    if (!ctx) { chartsBuilt = true; return; }
    var gctx = ctx.getContext('2d');
    var w = Math.max(ctx.parentElement.clientWidth || 600, 400);
    var grad = gctx.createLinearGradient(0, 0, w, 0);
    grad.addColorStop(0, c.lgRedDeep);
    grad.addColorStop(1, c.lgRed);
    var gray = gctx.createLinearGradient(0, 0, w, 0);
    gray.addColorStop(0, '#4A4A52');
    gray.addColorStop(1, '#8A8A92');
    barsChart = new Chart(gctx, {
      type: 'bar',
      data: {
        labels: ['탐색·이해', '설계·계획', '구현·검증', '배포·운영'],
        datasets: [{
          label: '%',
          data: [68, 45, 82, 37],
          backgroundColor: [grad, gray, grad, gray],
          borderRadius: 5,
          borderSkipped: false,
          barThickness: 22,
        }]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        layout: { padding: { top: 8, right: 24, bottom: 8, left: 8 } },
        plugins: {
          legend: { display: false },
          tooltip: {
            enabled: true,
            backgroundColor: c.text,
            titleColor: c.surface,
            bodyColor: c.surface,
            padding: 10,
            cornerRadius: 6,
            displayColors: false,
            callbacks: { label: function (ctx) { return ctx.parsed.x + '%'; } }
          }
        },
        scales: {
          x: {
            beginAtZero: true,
            max: 100,
            ticks: { color: c.textSecondary, font: { size: 10, family: 'Inter', weight: '600' }, callback: function (v) { return v + '%'; } },
            grid: { color: c.border, drawBorder: false },
            border: { display: false }
          },
          y: {
            ticks: { color: c.text, font: { size: 11, family: 'Inter', weight: '700' } },
            grid: { display: false },
            border: { display: false }
          }
        }
      }
    });
    chartsBuilt = true;
  };
  window.onThemeChange = function () {
    chartsBuilt = false;
    setTimeout(window.buildCharts, 120);
  };

  // ===== Print hooks =====
  window.addEventListener('beforeprint', function () {
    // 인쇄 전 카운터·애니메이션 즉시 완료 + 차트 리빌드
    document.querySelectorAll('[data-count]').forEach(function (el) {
      if (el.dataset.counted !== '1') {
        var target = parseFloat(el.dataset.count);
        var isFloat = String(target).indexOf('.') > -1;
        el.textContent = (el.dataset.prefix || '') + (isFloat ? target.toFixed(1) : Math.round(target).toLocaleString()) + (el.dataset.suffix || '');
        el.dataset.counted = '1';
      }
    });
    document.querySelectorAll('.animate-in').forEach(function (el) { el.classList.add('is-visible'); });
    chartsBuilt = false;
    window.buildCharts();
  });
  window.addEventListener('afterprint', function () {
    chartsBuilt = false;
    setTimeout(window.buildCharts, 100);
  });

  // ===== Init =====
  window.addEventListener('load', function () {
    window.buildTOC();
    window.buildDrawerTOC();
    window.numberFootnotes();
    setupFloatingControls();
    setupObserver();
    window.buildCharts();
  });
})();
