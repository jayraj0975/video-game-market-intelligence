const $ = (id) => document.getElementById(id);
const pct = (x, d = 1) => (x * 100).toFixed(d) + "%";
let slate = [];

async function api(path, body) {
  const res = await fetch(path, body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : undefined);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const d = data.detail;
    throw new Error(Array.isArray(d) ? d.map((e) => e.loc.slice(-1)[0] + ": " + e.msg).join("; ") : d || res.statusText);
  }
  return data;
}

function fill(sel, values, def) {
  $(sel).innerHTML = values.map((v) => `<option${v === def ? " selected" : ""}>${v}</option>`).join("");
}

function readForm() {
  const num = (id) => ($(id).value === "" ? null : Number($(id).value));
  return {
    title: $("title").value.trim(), platform: $("platform").value, genre: $("genre").value,
    rating: $("rating").value, publisher: $("publisher").value.trim(), year: Number($("year").value),
    critic_score: num("critic_score"), critic_count: num("critic_count"), n_platforms: Number($("n_platforms").value),
  };
}

function esc(s) { const d = document.createElement("div"); d.textContent = s; return d.innerHTML; }

function showResult(r) {
  const verdict = r.probability >= 0.5 ? "Likely hit" : r.probability >= r.base_rate * 1.5 ? "Above average odds" : "Below average odds";
  $("result").innerHTML = `
    <div class="big">${pct(r.probability)}</div>
    <div class="muted">chance of selling 1M+ &middot; ${verdict}</div>
    <div class="meter"><div style="width:${Math.min(100, r.probability * 100)}%"></div></div>
    <div class="muted" style="font-size:13px">Average release: ${pct(r.base_rate)} &middot; this one is ${r.vs_base_rate.toFixed(1)}&times; that</div>
    <div style="margin-top:14px">
      <div class="stat"><span>Publisher track record</span><span>${r.publisher_history.hits} hits / ${r.publisher_history.titles} titles</span></div>
      <div class="stat"><span>Franchise track record</span><span>${r.franchise_history.hits} hits / ${r.franchise_history.titles} titles</span></div>
      <div class="stat"><span>Critic data used</span><span>${r.used_critic_data ? "yes (near-launch estimate)" : "no (less reliable)"}</span></div>
      <div class="stat"><span>Decision point</span><span>${esc(r.decision_point)}</span></div>
    </div>
    ${r.history_note ? `<p class="caveat" role="note">${esc(r.history_note)}</p>` : ""}
    <p class="caveat">Physical retail sales only, and franchises are matched by the text before the colon in the title. A game already in the dataset counts itself in its own franchise history, so score new releases, not old ones. A single probability carries real uncertainty; the model is best at ordering titles against each other.</p>
    <h2 style="margin-top:18px">Similar past releases</h2>
    <div id="comparables"><p class="muted">Loading&hellip;</p></div>
    <h2 style="margin-top:18px">What would change the call?</h2>
    <div id="sensitivity"><p class="muted">Loading&hellip;</p></div>`;
}

const DIM_LABEL = { platform: "Platform", genre: "Genre", rating: "ESRB rating", n_platforms: "Platforms at launch" };

function showSensitivity(s) {
  const max = Math.max(...Object.values(s.dimensions).flatMap((d) => d.options.map((o) => o.probability)));
  $("sensitivity").innerHTML = Object.entries(s.dimensions).map(([dim, d]) => `
    <h3 class="sens-head">${DIM_LABEL[dim]} <span class="muted">(top ${Math.min(5, d.considered)} of ${d.considered})</span></h3>
    <div class="bars sens">${d.options.map((o) => `<div class="bar${o.current ? " now" : ""}"><span>${esc(String(o.value))}${o.current ? " (now)" : ""}</span>` +
      `<div class="track"><div class="fill" style="width:${(o.probability / max) * 100}%"></div></div>` +
      `<span class="num">${pct(o.probability)}${o.current ? "" : ` <small>${o.ratio.toFixed(1)}&times;</small>`}</span></div>`).join("")}</div>`).join("") +
    `<p class="caveat">${esc(s.note)} Only alternatives with enough past releases are shown.</p>`;
}

function loadExtras(form) {
  api("/api/comparables", form).then((c) => showComparables(c.comparables))
    .catch((err) => { $("comparables").innerHTML = `<p class="err">${esc(err.message)}</p>`; });
  api("/api/sensitivity", form).then(showSensitivity)
    .catch((err) => { $("sensitivity").innerHTML = `<p class="err">${esc(err.message)}</p>`; });
}

function showComparables(rows) {
  $("comparables").innerHTML = rows.length
    ? `<table><thead><tr><th>Title</th><th>Platform</th><th>Year</th><th class="num">Critic</th><th class="num">Sales (M)</th><th>Outcome</th></tr></thead><tbody>${
        rows.map((r) => `<tr><td>${esc(r.title)}</td><td>${esc(r.platform)}</td><td>${r.year}</td>` +
          `<td class="num">${r.critic_score != null ? r.critic_score : "&ndash;"}</td>` +
          `<td class="num">${r.global_sales_munits.toFixed(2)}</td>` +
          `<td>${r.is_hit ? "Hit (1M+)" : "Did not hit"}</td></tr>`).join("")
      }</tbody></table><p class="caveat">Nearest past releases by platform, genre, rating, publisher and critic score &mdash; not a model input, just what actually happened to similar titles.</p>`
    : `<p class="muted">No comparable past release shares enough with this one to be useful.</p>`;
}

function showConceptReport(r) {
  const sat = r.market_saturation;
  const risks = r.risk_factors.length
    ? `<ul>${r.risk_factors.map((f) => `<li>${esc(f)}</li>`).join("")}</ul>`
    : `<p class="muted">No specific risk flags from this dataset's history.</p>`;
  $("concept-report").innerHTML = `
    <h2 style="margin-top:18px">Concept validation</h2>
    <div class="stat"><span>${esc(sat.recent_window)} releases, this genre+platform</span><span>${sat.same_genre_and_platform_recent_releases} of ${sat.same_genre_and_platform_releases} total</span></div>
    <div class="stat"><span>Hit rate, this genre+platform</span><span>${sat.same_genre_and_platform_hit_rate != null ? pct(sat.same_genre_and_platform_hit_rate) : "no history"}</span></div>
    <div class="stat"><span>Price distribution</span><span class="muted">${esc(r.price_distribution_note)}</span></div>
    <h3 style="margin:14px 0 6px; font-size:14px">Risk factors</h3>
    ${risks}
    <p class="caveat">${esc(sat.note)}</p>`;
  showResult(r.prediction);
  showComparables(r.comparables);
  api("/api/sensitivity", readForm()).then(showSensitivity)
    .catch((err) => { $("sensitivity").innerHTML = `<p class="err">${esc(err.message)}</p>`; });
}

$("validate-concept").addEventListener("click", async () => {
  $("err").textContent = ""; $("concept-report").innerHTML = "";
  try { showConceptReport(await api("/api/concept-validate", readForm())); }
  catch (err) { $("err").textContent = err.message; }
});

$("form").addEventListener("submit", async (e) => {
  e.preventDefault(); $("err").textContent = "";
  try {
    const form = readForm();
    showResult(await api("/api/predict", form));
    loadExtras(form);
  } catch (err) { $("err").textContent = err.message; }
});

function renderSlate() {
  $("slate").innerHTML = slate.length
    ? `<table><thead><tr><th>#</th><th>Title</th><th>Platform</th><th>Genre</th><th class="num">Hit chance</th></tr></thead><tbody>${
        slate.map((r, i) => `<tr><td>${r.rank ?? i + 1}</td><td>${esc(r.title || "(untitled)")}</td><td>${esc(r.platform)}</td><td>${esc(r.genre)}</td><td class="num">${r.probability != null ? pct(r.probability) : "&ndash;"}</td></tr>`).join("")
      }</tbody></table>`
    : `<p class="muted">Nothing added yet.</p>`;
}

$("add").addEventListener("click", () => {
  $("err").textContent = "";
  if (slate.length >= 50) { $("err").textContent = "The slate holds at most 50 releases."; return; }
  slate.push(readForm()); renderSlate();
});
$("clear").addEventListener("click", () => { slate = []; renderSlate(); });
$("rank").addEventListener("click", async () => {
  if (!slate.length) return;
  try {
    const out = await api("/api/rank", { releases: slate.map(({ rank, probability, ...rest }) => rest) });
    // Keep the original inputs so the slate can be edited and re-ranked.
    const inputs = slate.map(({ rank, probability, ...rest }) => rest);
    const used = new Set();
    slate = out.ranked.map((r) => {
      const i = inputs.findIndex((x, j) => !used.has(j) && x.title === r.title && x.platform === r.platform && x.genre === r.genre);
      used.add(i);
      return { ...inputs[i], probability: r.probability, rank: r.rank };
    });
    renderSlate();
  } catch (err) { $("err").textContent = err.message; }
});

function bars(el, rows, fmt) {
  const max = Math.max(...rows.map((r) => r.value));
  $(el).innerHTML = rows.map((r) => `<div class="bar"><span>${esc(r.label)}</span><div class="track"><div class="fill" style="width:${(r.value / max) * 100}%"></div></div><span class="num">${fmt(r.value)}</span></div>`).join("");
}

async function init() {
  const [opt, market, metrics, info] = await Promise.all([api("/api/options"), api("/api/market"), api("/api/metrics"), api("/api/model-info").catch(() => null)]);
  const abl = metrics.ablation;
  if (abl && abl.length === 2) {
    $("critic-hint").textContent = `Critic score and review count go together, and only exist close to launch. Leave both blank for an estimate without them; it is less reliable (on the held-out 2014\u201315 slate, a model trained without critic features scores PR-AUC ${abl[1].pr_auc.toFixed(3)}, against ${abl[0].pr_auc.toFixed(3)} with them).`;
  }
  if (info) {
    $("provenance").textContent = `Model v${info.model_version} \u00b7 ${info.decision_point} \u00b7 trained on ${info.training_window}, evaluated on ${info.evaluation_window.split(" ")[0]} \u00b7 commit ${(info.code_commit || "unknown").slice(0, 7)} \u00b7 data SHA-256 ${info.data_sha256.clean_csv.slice(0, 12)}\u2026 \u00b7 artifact SHA-256 ${info.artifact_sha256.slice(0, 12)}\u2026 \u00b7 scikit-learn ${info.scikit_learn}`;
  }
  fill("platform", opt.platforms, "PS3"); fill("genre", opt.genres, "Action"); fill("rating", opt.ratings, "T");
  $("publishers").innerHTML = opt.publishers.map((p) => `<option value="${esc(p)}">`).join("");

  const regions = Object.keys(market.regional_genre_share);
  const draw = (region) => {
    [...$("regions").children].forEach((b) => b.setAttribute("aria-pressed", b.textContent === region));
    const rows = Object.entries(market.regional_genre_share[region]).map(([label, value]) => ({ label, value })).sort((a, b) => b.value - a.value);
    bars("genre-bars", rows, (v) => pct(v));
  };
  $("regions").innerHTML = regions.map((r) => `<button type="button" aria-pressed="false">${r}</button>`).join("");
  [...$("regions").children].forEach((b) => b.addEventListener("click", () => draw(b.textContent)));
  draw("Japan");

  bars("score-bars", market.hit_rate_by_score.map((r) => ({ label: r.band, value: r.hit_rate })), (v) => pct(v, 0));

  const best = metrics.models.find((m) => m.model === metrics.best_model);
  const ci = metrics.bootstrap_95ci;
  $("metrics").innerHTML = `
    <div class="stat"><span>Model</span><span>${esc(metrics.best_model)}</span></div>
    <div class="stat"><span>PR-AUC (no skill ${metrics.base_rate.toFixed(2)})</span><span>${best.pr_auc.toFixed(3)}</span></div>
    <div class="stat"><span>Top 10% captures</span><span>${pct(best.top10_recall, 0)} of hits (${best.top10_lift.toFixed(1)}&times;)</span></div>
    <div class="stat"><span>Lift 95% interval</span><span>${ci.best_lift[0].toFixed(1)}&ndash;${ci.best_lift[1].toFixed(1)}&times;</span></div>`;
  renderSlate();
}
init().catch((e) => { $("err").textContent = "Could not load: " + e.message; });
