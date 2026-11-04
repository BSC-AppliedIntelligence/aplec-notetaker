// Gmail-"To"-style member picker: type to search the roster, Enter/click fixes
// a chip, × (or Backspace on empty input) removes it. Selected values ride as
// hidden inputs (name from data-name), so forms submit the same shape the old
// checkboxes did. Vanilla JS, DOM built via textContent only.
(function () {
  function init(box) {
    if (box.dataset.chipsReady) return;
    box.dataset.chipsReady = "1";
    const roster = JSON.parse(box.dataset.roster || "[]");
    const fieldName = box.dataset.name || "members";
    const selected = [];
    const chips = box.querySelector(".chips-selected");
    const input = box.querySelector(".chips-input");
    const dd = box.querySelector(".chips-dd");
    const hidden = box.querySelector(".chips-hidden");
    let items = [];
    let active = -1;

    function syncHidden() {
      hidden.textContent = "";
      selected.forEach((email) => {
        const h = document.createElement("input");
        h.type = "hidden";
        h.name = fieldName;
        h.value = email;
        hidden.appendChild(h);
      });
    }

    function addChip(email) {
      if (selected.includes(email)) return;
      selected.push(email);
      const chip = document.createElement("span");
      chip.className = "chip";
      const label = document.createElement("span");
      label.textContent = email;
      const x = document.createElement("button");
      x.type = "button";
      x.className = "chip-x";
      x.textContent = "×";
      x.setAttribute("aria-label", (box.dataset.removeLabel || "Remove") + " " + email);
      x.addEventListener("click", () => {
        selected.splice(selected.indexOf(email), 1);
        chip.remove();
        syncHidden();
        input.focus();
      });
      chip.appendChild(label);
      chip.appendChild(x);
      chips.appendChild(chip);
      syncHidden();
      input.value = "";
      closeDd();
    }

    function closeDd() {
      dd.hidden = true;
      dd.textContent = "";
      items = [];
      active = -1;
    }

    function renderDd() {
      const q = input.value.trim().toLowerCase();
      const matches = roster
        .filter((e) => !selected.includes(e) && (!q || e.toLowerCase().includes(q)))
        .slice(0, 8);
      dd.textContent = "";
      items = [];
      active = matches.length ? 0 : -1;
      if (!matches.length) {
        dd.hidden = true;
        return;
      }
      matches.forEach((email, i) => {
        const item = document.createElement("button");
        item.type = "button";
        item.className = "chips-dd-item";
        item.textContent = email;
        item.addEventListener("mousedown", (ev) => {
          ev.preventDefault(); // beat the input's blur
          addChip(email);
        });
        item.addEventListener("mousemove", () => setActive(i));
        dd.appendChild(item);
        items.push(item);
      });
      setActive(active);
      dd.hidden = false;
    }

    function setActive(i) {
      active = i;
      items.forEach((el, j) => el.classList.toggle("active", j === active));
    }

    input.addEventListener("input", renderDd);
    input.addEventListener("focus", renderDd);
    input.addEventListener("blur", () => setTimeout(closeDd, 150));
    input.addEventListener("keydown", (ev) => {
      if (ev.key === "ArrowDown" && items.length) {
        ev.preventDefault();
        setActive((active + 1) % items.length);
      } else if (ev.key === "ArrowUp" && items.length) {
        ev.preventDefault();
        setActive((active - 1 + items.length) % items.length);
      } else if (ev.key === "Enter" && !dd.hidden && active >= 0) {
        ev.preventDefault(); // choose, don't submit
        addChip(items[active].textContent);
      } else if (ev.key === "Escape") {
        closeDd();
      } else if (ev.key === "Backspace" && !input.value && selected.length) {
        selected.pop();
        chips.lastElementChild.remove();
        syncHidden();
      }
    });

    JSON.parse(box.dataset.selected || "[]").forEach(addChip);
  }

  function initAll() {
    document.querySelectorAll(".chips-box").forEach(init);
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initAll);
  } else {
    initAll();
  }
})();
