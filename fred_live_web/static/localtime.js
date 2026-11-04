// Dates and times in the VIEWER's time zone.
//
// The server renders every time in its own zone (the stack's TZ) as a
// fallback, inside <time data-lt="time|day|date|datetime" datetime="UTC ISO">,
// and this rewrites them in the browser's zone. The meetings list is regrouped
// by the viewer's day as well: a call at 00:30 in Madrid is still
// "Yesterday" in São Paulo. Runs on load and on every htmx swap (search,
// filters, the 12 s poll while a meeting is processing).
(function () {
  // The page's interface language (<html lang>, set from Settings). Same words
  // and order as the server's fmt_day (fred_live_web/i18n.py format_day);
  // weekdays here start on Sunday, as getDay() does.
  const LANGS = {
    en: {
      today: "Today", yesterday: "Yesterday",
      weekdays: ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"],
      months: ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
      near: (w, d, m) => `${w}, ${m} ${d}`, far: (d, m, y) => `${m} ${d}, ${y}`,
    },
    es: {
      today: "Hoy", yesterday: "Ayer",
      weekdays: ["Dom", "Lun", "Mar", "Mié", "Jue", "Vie", "Sáb"],
      months: ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sept", "oct", "nov", "dic"],
      near: (w, d, m) => `${w}, ${d} ${m}`, far: (d, m, y) => `${d} ${m} ${y}`,
    },
    ca: {
      today: "Avui", yesterday: "Ahir",
      weekdays: ["Dg.", "Dl.", "Dt.", "Dc.", "Dj.", "Dv.", "Ds."],
      months: ["de gen.", "de febr.", "de març", "d’abr.", "de maig", "de juny", "de jul.",
               "d’ag.", "de set.", "d’oct.", "de nov.", "de des."],
      near: (w, d, m) => `${w}, ${d} ${m}`, far: (d, m, y) => `${d} ${m} de ${y}`,
    },
  };
  const L = LANGS[(document.documentElement.lang || "en").slice(0, 2)] || LANGS.en;
  const pad = (n) => String(n).padStart(2, "0");

  function parse(iso) {
    const d = iso ? new Date(iso) : null;
    return d && !isNaN(d) ? d : null;
  }

  // Same rules and wording as the server's fmt_day / fmt_time.
  function day(d) {
    const midnight = (x) => new Date(x.getFullYear(), x.getMonth(), x.getDate());
    const delta = Math.round((midnight(new Date()) - midnight(d)) / 86400000);
    if (delta === 0) return L.today;
    if (delta === 1) return L.yesterday;
    return delta < 300
      ? L.near(L.weekdays[d.getDay()], d.getDate(), L.months[d.getMonth()])
      : L.far(d.getDate(), L.months[d.getMonth()], d.getFullYear());
  }
  const time = (d) => `${pad(d.getHours())}:${pad(d.getMinutes())}`;
  const date = (d) => `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
  const FORMATS = { day, time, date, datetime: (d) => `${date(d)} ${time(d)}` };

  function rewrite(root) {
    root.querySelectorAll("time[data-lt]").forEach((el) => {
      const d = parse(el.getAttribute("datetime"));
      const fmt = FORMATS[el.dataset.lt];
      if (d && fmt) el.textContent = fmt(d);
    });
  }

  // Cards arrive newest first, grouped by the SERVER's day; rebuild the
  // groups by the viewer's. Order never changes: an instant sorts the same in
  // every zone.
  function regroup() {
    const content = document.getElementById("meetings-content");
    if (!content) return;
    const groups = [...content.querySelectorAll(":scope > .day-group")];
    if (!groups.length) return;
    const cards = [];
    groups.forEach((group) => {
      const serverLabel = group.querySelector(".day-label")?.textContent || "";
      group.querySelectorAll(":scope > .meeting-card").forEach((card) => {
        const d = parse(card.dataset.start);
        cards.push({ card, label: d ? day(d) : serverLabel });
      });
    });
    const rebuilt = [];
    cards.forEach(({ card, label }) => {
      let section = rebuilt[rebuilt.length - 1];
      if (!section || section.dataset.day !== label) {
        section = document.createElement("section");
        section.className = "day-group";
        section.dataset.day = label;
        const h = document.createElement("h2");
        h.className = "day-label";
        h.textContent = label;
        section.appendChild(h);
        rebuilt.push(section);
      }
      section.appendChild(card);
    });
    groups[0].before(...rebuilt);
    groups.forEach((group) => group.remove());
  }

  function run(root) {
    rewrite(root || document);
    regroup();
  }

  // For the copy-notes button: the meeting's start in the viewer's zone.
  window.fredLocalTime = {
    when(iso) {
      const d = parse(iso);
      return d ? `${day(d)} ${time(d)}` : "";
    },
  };

  if (window.htmx) {
    window.htmx.onLoad(run);
  } else {
    document.addEventListener("DOMContentLoaded", () => run(document));
  }
})();
