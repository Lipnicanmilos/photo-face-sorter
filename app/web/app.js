"use strict";

const $ = (id) => document.getElementById(id);

let persons = [];
let pollTimer = null;

// ------------------------------------------------------------------- helpers

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.detail || `${response.status} ${response.statusText}`);
  }
  return payload;
}

let toastTimer = null;
function toast(message, kind = "ok") {
  const element = $("toast");
  element.textContent = message;
  element.className = `toast ${kind}`;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => element.classList.add("hidden"), kind === "err" ? 8000 : 4000);
}

function busy(isBusy) {
  for (const id of ["scan-btn", "cluster-btn", "sort-btn", "sort-preview"]) {
    $(id).disabled = isBusy;
  }
}

// -------------------------------------------------------------------- štatistiky

async function loadStats() {
  const stats = await api("/api/stats");
  $("stats-chips").innerHTML = [
    ["fotiek", stats.photos],
    ["tvárí", stats.faces],
    ["osôb", stats.persons],
    ["nepriradených", stats.unassigned_faces],
  ]
    .map(([label, value]) => `<span class="chip">${label} <b>${value}</b></span>`)
    .join("");

  if ($("cluster-eps").value === "") $("cluster-eps").value = stats.defaults.eps;
  if ($("cluster-min-samples").value === "") $("cluster-min-samples").value = stats.defaults.min_samples;
  if ($("sort-output").value === "") $("sort-output").value = stats.defaults.output_dir;
  if ($("sort-mode").dataset.init !== "1") {
    $("sort-mode").value = stats.defaults.link_mode;
    $("sort-mode").dataset.init = "1";
  }

  if (stats.active_job && stats.active_job.status === "running") {
    busy(true);
    watchJob(stats.active_job.id);
  }
  return stats;
}

// -------------------------------------------------------------------------- scan

async function startScan() {
  const folder = $("scan-folder").value.trim();
  if (!folder) return toast("Zadaj priečinok s fotografiami.", "err");

  try {
    busy(true);
    const job = await api("/api/scan", {
      method: "POST",
      body: JSON.stringify({
        folder,
        recursive: $("scan-recursive").checked,
        force: $("scan-force").checked,
      }),
    });
    $("scan-progress").classList.remove("hidden");
    $("progress-text").textContent = "Spúšťam…";
    watchJob(job.id);
  } catch (error) {
    busy(false);
    toast(error.message, "err");
  }
}

function watchJob(jobId) {
  clearInterval(pollTimer);
  $("scan-progress").classList.remove("hidden");

  pollTimer = setInterval(async () => {
    let job;
    try {
      job = await api(`/api/jobs/${jobId}`);
    } catch (error) {
      clearInterval(pollTimer);
      busy(false);
      return toast(error.message, "err");
    }

    const percent = job.total ? Math.round((job.current / job.total) * 100) : 0;
    $("progress-fill").style.width = `${percent}%`;
    $("progress-text").textContent = job.total
      ? `${job.current}/${job.total} — ${job.message}`
      : job.message;

    if (job.status === "running") return;

    clearInterval(pollTimer);
    busy(false);

    if (job.status === "error") {
      toast(job.error, "err");
      return;
    }

    const report = job.result || {};
    $("progress-fill").style.width = "100%";
    $("progress-text").textContent =
      `Hotovo: ${report.processed} nových (+${report.new_faces} tvárí), ` +
      `${report.skipped_cached} z cache, ${(report.failures || []).length} chýb, ` +
      `${(report.elapsed_seconds || 0).toFixed(1)} s`;
    toast(`Naskenované: ${report.processed} nových fotiek, ${report.new_faces} tvárí.`);
    await loadStats();
    await loadPersons();
  }, 700);
}

// -------------------------------------------------------------------- zhlukovanie

async function runCluster() {
  try {
    busy(true);
    const result = await api("/api/cluster", {
      method: "POST",
      body: JSON.stringify({
        eps: parseFloat($("cluster-eps").value) || null,
        min_samples: parseInt($("cluster-min-samples").value, 10) || null,
      }),
    });
    toast(`Rozpoznaných ${result.persons} osôb, ${result.unassigned} tvárí nepriradených.`);
    await loadStats();
    await loadPersons();
  } catch (error) {
    toast(error.message, "err");
  } finally {
    busy(false);
  }
}

// -------------------------------------------------------------------------- osoby

async function loadPersons() {
  const data = await api("/api/persons");
  persons = data.persons;
  const grid = $("persons-grid");
  grid.innerHTML = "";

  $("persons-empty").classList.toggle("hidden", persons.length > 0 || data.unassigned > 0);

  for (const person of persons) {
    grid.appendChild(personCard(person));
  }
  if (data.unassigned > 0) {
    grid.appendChild(unassignedCard(data.unassigned));
  }
}

function personCard(person) {
  const card = document.createElement("div");
  card.className = "person";

  const preview = person.preview_file
    ? `<img class="person-preview" src="/faces/${person.preview_file}" alt="${person.label}" loading="lazy">`
    : `<div class="person-preview placeholder">🙂</div>`;

  card.innerHTML = `
    ${preview}
    <div class="person-body">
      <input class="person-name" value="${escapeAttr(person.display_name || "")}"
             placeholder="${person.label}" title="Klikni a napíš meno">
      <div class="person-meta">${person.face_count} tvárí · ${person.photo_count} fotiek</div>
      <div class="person-actions">
        <select title="Zlúčiť túto osobu do inej">
          <option value="">zlúčiť do…</option>
          ${persons
            .filter((other) => other.label !== person.label)
            .map((other) => `<option value="${other.label}">${escapeHtml(other.folder_name)}</option>`)
            .join("")}
        </select>
      </div>
    </div>`;

  card.querySelector(".person-preview").addEventListener("click", () => showFaces(person.label, person.folder_name));

  const nameInput = card.querySelector(".person-name");
  nameInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter") nameInput.blur();
  });
  nameInput.addEventListener("blur", async () => {
    const value = nameInput.value.trim();
    if (value === (person.display_name || "")) return;
    try {
      await api(`/api/persons/${encodeURIComponent(person.label)}`, {
        method: "PATCH",
        body: JSON.stringify({ display_name: value }),
      });
      toast(`${person.label} → ${value || person.label}`);
      await loadPersons();
    } catch (error) {
      toast(error.message, "err");
    }
  });

  card.querySelector("select").addEventListener("change", async (event) => {
    const target = event.target.value;
    if (!target) return;
    if (!confirm(`Zlúčiť „${person.folder_name}" do „${target}"?`)) {
      event.target.value = "";
      return;
    }
    try {
      await api("/api/persons/merge", {
        method: "POST",
        body: JSON.stringify({ source: person.label, target }),
      });
      toast("Osoby zlúčené.");
      await loadStats();
      await loadPersons();
    } catch (error) {
      toast(error.message, "err");
    }
  });

  return card;
}

function unassignedCard(count) {
  const card = document.createElement("div");
  card.className = "person unassigned";
  card.innerHTML = `
    <div class="person-preview placeholder">❔</div>
    <div class="person-body">
      <div class="person-name" style="font-weight:600">Nepriradené</div>
      <div class="person-meta">${count} tvárí</div>
    </div>`;
  card.querySelector(".person-preview").addEventListener("click", () => showFaces("_unassigned", "Nepriradené tváre"));
  return card;
}

async function showFaces(label, title) {
  try {
    const data = await api(`/api/persons/${encodeURIComponent(label)}/faces`);
    $("modal-title").textContent = `${title} — ${data.faces.length} tvárí`;
    $("modal-faces").innerHTML = data.faces
      .map(
        (face) => `
        <div class="face">
          <img src="/faces/${face.preview_file}" loading="lazy"
               data-path="${escapeAttr(face.source_path)}" title="Otvoriť pôvodnú fotku">
          <span title="${escapeAttr(face.source_path)}">${escapeHtml(fileName(face.source_path))}</span>
        </div>`
      )
      .join("");
    for (const img of $("modal-faces").querySelectorAll("img")) {
      img.addEventListener("click", () => {
        window.open(`/api/photo?path=${encodeURIComponent(img.dataset.path)}`, "_blank");
      });
    }
    $("modal").classList.remove("hidden");
  } catch (error) {
    toast(error.message, "err");
  }
}

// ---------------------------------------------------------------------- triedenie

async function runSort(dryRun) {
  try {
    busy(true);
    const report = await api("/api/sort", {
      method: "POST",
      body: JSON.stringify({
        output: $("sort-output").value.trim() || null,
        mode: $("sort-mode").value,
        dry_run: dryRun,
        clean: $("sort-clean").checked,
      }),
    });
    renderSortReport(report);
    toast(
      dryRun
        ? `Plán: ${report.total_photos} položiek v ${Object.keys(report.folders).length} priečinkoch.`
        : `Roztriedené: ${report.linked} položiek.`
    );
  } catch (error) {
    toast(error.message, "err");
  } finally {
    busy(false);
  }
}

function renderSortReport(report) {
  const folders = Object.entries(report.folders);
  if (!folders.length) {
    $("sort-result").innerHTML = `<p class="summary">Niet čo triediť.</p>`;
    return;
  }

  const rows = folders
    .map(([name, count]) => `<tr><td>${escapeHtml(name)}</td><td class="num">${count}</td></tr>`)
    .join("");

  const extra = report.dry_run
    ? `<p class="summary">Náhľad — na disk sa nič nezapísalo.</p>`
    : `<p class="summary">
         Vytvorených <b>${report.linked}</b> položiek, už existovalo <b>${report.skipped_existing}</b>
         ${report.fallback_copies ? `, kópií namiesto linku <b>${report.fallback_copies}</b>` : ""}.
       </p>`;

  const errors = (report.errors || []).length
    ? `<p class="summary">Chyby: ${report.errors.length} — prvá: ${escapeHtml(report.errors[0][1])}</p>`
    : "";

  $("sort-result").innerHTML = `
    <table class="report">
      <thead><tr><th>Priečinok</th><th class="num">Fotiek</th></tr></thead>
      <tbody>${rows}</tbody>
      <tfoot><tr class="total"><td>Spolu</td><td class="num">${report.total_photos}</td></tr></tfoot>
    </table>${extra}${errors}`;
}

// ------------------------------------------------------------------------ utils

const fileName = (path) => path.split(/[\\/]/).pop();
const escapeHtml = (text) =>
  String(text).replace(/[&<>]/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" })[char]);
const escapeAttr = (text) => escapeHtml(text).replace(/"/g, "&quot;");

// ------------------------------------------------------------------------ štart

$("scan-btn").addEventListener("click", startScan);
$("cluster-btn").addEventListener("click", runCluster);
$("refresh-persons").addEventListener("click", () => loadPersons().catch((e) => toast(e.message, "err")));
$("sort-preview").addEventListener("click", () => runSort(true));
$("sort-btn").addEventListener("click", () => runSort(false));
$("modal-close").addEventListener("click", () => $("modal").classList.add("hidden"));
$("modal").addEventListener("click", (event) => {
  if (event.target === $("modal")) $("modal").classList.add("hidden");
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") $("modal").classList.add("hidden");
});
$("scan-folder").addEventListener("keydown", (event) => {
  if (event.key === "Enter") startScan();
});

loadStats().then(loadPersons).catch((error) => toast(error.message, "err"));
