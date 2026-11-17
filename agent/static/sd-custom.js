// "Check any Hugging Face repo" box for the image/video section of the Models panel.
// Load with <script src="static/sd-custom.js" defer></script>. Added models appear in the
// existing preset list (Download, then Use); the fit badge is already in the model name.
(() => {
  if (window.__buiaSdCustom) return;
  window.__buiaSdCustom = true;
  const api = (p, b) => fetch("/api/sd/" + p, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(b)})
    .then(async r => { const j = await r.json().catch(() => ({})); if (!r.ok) throw new Error(j.detail || r.statusText); return j; });
  const box = document.createElement("div");
  box.style.cssText = "margin:8px 0;padding:8px;border:1px solid var(--border,#8884);border-radius:8px";
  box.innerHTML = '<b>Add an image/video model from Hugging Face</b>' +
    '<div style="display:flex;gap:6px;margin-top:6px"><input class="sd-repo" placeholder="owner/repo" style="flex:1">' +
    '<button class="sd-check">Check</button></div><div class="sd-out" style="margin-top:6px"></div>';
  let mounted = false;
  const mount = () => {
    if (mounted) return true;
    const h = [...document.querySelectorAll("h1,h2,h3,h4,summary,b,strong,div,span")]
      .find(e => /Image and video models/i.test(e.textContent || "") && e.children.length <= 2);
    if (!h) return false;
    (h.closest("section,details,div") || h.parentElement || document.body).insertAdjacentElement("afterend", box);
    mounted = true;
    return true;
  };
  if (!mount()) {
    const obs = new MutationObserver(() => { if (mount()) obs.disconnect(); });
    obs.observe(document.documentElement, {childList: true, subtree: true});
  }
  box.querySelector(".sd-check").onclick = async () => {
    const out = box.querySelector(".sd-out");
    out.textContent = "checking...";
    try {
      const j = await api("check", {repo: box.querySelector(".sd-repo").value});
      out.innerHTML = "";
      for (const f of j.families) for (const v of f.variants) {
        const row = document.createElement("div");
        row.style.cssText = "display:flex;gap:6px;align-items:center;margin:4px 0";
        const word = {fits: "fits", tight: "tight", offload: "needs RAM offload", no: "may not fit"};
        const fit = v.fit ? (word[v.fit.verdict] || "fit unknown") + " " + Math.round(v.fit.vram_mb / 1024) + " GB" : "fit unknown";
        const label = document.createElement("span");
        label.style.flex = "1";
        label.textContent = f.name + " - " + v.path + " (" + v.size_mb + " MB" + (v.detail ? ", " + v.detail : "") + ") ";
        const i = document.createElement("i");
        i.textContent = fit;
        if (v.fit) i.title = v.fit.reason + " Estimated for " + v.fit.ref + " (" + v.fit.basis + (v.fit.source ? ", weights from " + v.fit.source : "") + ").";
        label.appendChild(i);
        if (v.warning) {
          const w = document.createElement("div");
          w.textContent = v.warning;
          w.style.cssText = "font-size:.85em;opacity:.8";
          label.appendChild(w);
        }
        row.appendChild(label);
        const b = document.createElement("button");
        b.textContent = "Add";
        b.onclick = async () => {
          b.disabled = true;
          try {
            await api("add", {repo: j.repo, path: v.path});
            b.textContent = "Added";
            window.dispatchEvent(new CustomEvent("buia:sd-changed"));
          } catch (e) { b.disabled = false; out.textContent = "error: " + e.message; }
        };
        row.appendChild(b);
        out.appendChild(row);
      }
      if (j.skipped) {
        const n = document.createElement("div");
        n.textContent = j.skipped + " more large .safetensors files were not checked (the largest 12 are read).";
        out.appendChild(n);
      }
    } catch (e) { out.textContent = "error: " + e.message; }
  };
})();
