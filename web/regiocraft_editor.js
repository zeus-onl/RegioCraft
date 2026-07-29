// RegioCraft — in-node visual region editor + per-region control rows.
// Combines:
//   - the draggable/resizable box-canvas from regional_character_lora
//     (generalized here from 2 fixed boxes to N boxes, rainbow-colored)
//   - the per-row widgets + ref-image upload/thumbnail from Fedor's
//     Krea2-Multi-Character-Lora-Node v3 (generalized the same way)
//
// The canvas IS the source of truth for box geometry (x/y/w/h, normalized
// 0-1); the per-row widgets are the source of truth for lora/strength/enable/
// ref_image. Both write into the same "regions_json" array so the Python
// node's "manual" split_mode can read x/y/w/h straight off each region.
//
// Credits: this editor merges UI work from Fedor's Krea2-Multi-Character-
// Lora-Node-w-bounding-box (per-row lora/strength/ref widgets, ref-image
// upload+thumbnail) and Shy's fork of Gorecheese's original
// regional_character_lora (the draggable box canvas, the "load latest
// output as background" workflow). Thanks to all three.

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_TYPE = "RegioCraft";
const JSON_WIDGET = "regions_json";
const HANDLE = 12;
const THUMB_H = 48;
const MINSIZE = 0.04;

let LORA_LIST = ["None"];

async function ensureLoraList() {
  if (LORA_LIST.length > 1) return LORA_LIST;
  try {
    const resp = await api.fetchApi("/object_info/LoraLoader");
    const info = await resp.json();
    const names = info?.LoraLoader?.input?.required?.lora_name?.[0];
    if (Array.isArray(names) && names.length) {
      LORA_LIST = ["None", ...names.filter((n) => n !== "None")];
    }
  } catch (e) {
    console.warn("[RegioCraft] could not fetch lora list:", e);
  }
  return LORA_LIST;
}

async function fetchLatestOutputImageURL() {
  // Walks ComfyUI's /history from most-recent backwards and returns a /view
  // URL for the first image output it finds -- used by "Load latest output"
  // as a one-shot pull.
  try {
    const res = await fetch("/history?max_items=30");
    if (!res.ok) return null;
    const hist = await res.json();
    const entries = Object.values(hist);
    for (let i = entries.length - 1; i >= 0; i--) {
      const outputs = (entries[i] && entries[i].outputs) || {};
      for (const nodeId of Object.keys(outputs)) {
        const imgs = outputs[nodeId].images;
        if (imgs && imgs.length) {
          const img = imgs[imgs.length - 1];
          const params = new URLSearchParams({
            filename: img.filename, subfolder: img.subfolder || "", type: img.type || "output",
          });
          return "/view?" + params.toString();
        }
      }
    }
  } catch (e) {
    console.error("[RegioCraft] failed to fetch latest output:", e);
  }
  return null;
}

function imageOutputURLFromExecutedEvent(detail) {
  const imgs = detail && detail.output && detail.output.images;
  if (!imgs || !imgs.length) return null;
  const img = imgs[imgs.length - 1];
  const params = new URLSearchParams({
    filename: img.filename, subfolder: img.subfolder || "", type: img.type || "output",
  });
  return "/view?" + params.toString();
}

function hueColor(i, n, alpha = 1) {
  const hue = (i / Math.max(1, n)) * 300; // stop before wrapping back to red
  return `hsla(${hue}, 70%, 60%, ${alpha})`;
}

function defaultRegion(i, n) {
  const cols = Math.max(1, n);
  return {
    name: `region_${i}`,
    lora: "None",
    strength: 1.2,
    enable: true,
    ref_image: "",
    x: i / cols, y: 0.0, w: 1.0 / cols, h: 1.0,
  };
}

function defaultRegions() {
  return [defaultRegion(0, 2), defaultRegion(1, 2)];
}

function clamp01(v) { return Math.max(0, Math.min(1, v)); }

function readRegions(node) {
  const w = node.widgets?.find((x) => x.name === JSON_WIDGET);
  if (!w) return [];
  try {
    const parsed = JSON.parse(w.value || "[]");
    return Array.isArray(parsed) ? parsed : [];
  } catch (e) {
    return [];
  }
}

function writeRegions(node, regions) {
  const w = node.widgets?.find((x) => x.name === JSON_WIDGET);
  if (!w) return;
  w.value = JSON.stringify(regions, null, 2);
  if (w.inputEl) w.inputEl.value = w.value;
}

function markTransient(w) {
  w.__rc_row = true;
  w.serialize = false;
  if (!w.options) w.options = {};
  w.options.serialize = false;
  return w;
}

function shortName(p) {
  if (!p || typeof p !== "string" || p === "None") return "";
  const s = p.split(/[\\/]/).pop().replace(/\.safetensors$/i, "");
  return s.length > 14 ? s.slice(0, 13) + "…" : s;
}

// ---------------------------------------------------------------------------
// reference image upload + thumbnail cache (from Fedor's v3 widget, unchanged idea)
// ---------------------------------------------------------------------------
const THUMB_CACHE = {};

function thumbFor(name, node) {
  if (!name) return null;
  let img = THUMB_CACHE[name];
  if (!img) {
    img = new Image();
    const slash = name.lastIndexOf("/");
    const subfolder = slash >= 0 ? name.slice(0, slash) : "";
    const fname = slash >= 0 ? name.slice(slash + 1) : name;
    img.src = api.apiURL(
      `/view?filename=${encodeURIComponent(fname)}&type=input&subfolder=${encodeURIComponent(subfolder)}&rand=${Math.random()}`
    );
    img.onload = () => node.setDirtyCanvas(true, true);
    THUMB_CACHE[name] = img;
  }
  return img.complete && img.naturalWidth ? img : null;
}

async function uploadRefImage(file) {
  const body = new FormData();
  body.append("image", file);
  body.append("type", "input");
  const resp = await api.fetchApi("/upload/image", { method: "POST", body });
  if (resp.status !== 200) {
    console.error("[RegioCraft] ref upload failed:", resp.status);
    return null;
  }
  const data = await resp.json();
  return data.subfolder ? `${data.subfolder}/${data.name}` : data.name;
}

function pickAndUploadRef(node, idx) {
  const input = document.createElement("input");
  input.type = "file";
  input.accept = "image/png,image/jpeg,image/webp,image/bmp";
  input.onchange = async () => {
    const file = input.files?.[0];
    if (!file) return;
    const name = await uploadRefImage(file);
    if (!name) return;
    const r = readRegions(node);
    if (r[idx]) {
      r[idx].ref_image = name;
      writeRegions(node, r);
      delete THUMB_CACHE[name];
      rebuildRows(node);
    }
  };
  input.click();
}

function makeRefWidget(node, idx, region) {
  const w = {
    type: "RC_REF",
    name: `region ${idx + 1} ref`,
    value: region.ref_image || "",
    serialize: false,
    options: { serialize: false },
    computeSize(width) {
      return [width, this.value ? THUMB_H + 8 : 20];
    },
    draw(ctx, drawNode, widgetWidth, y) {
      const margin = 12;
      const w_ = widgetWidth - margin * 2;
      ctx.save();
      ctx.fillStyle = "#353535";
      ctx.strokeStyle = "#555";
      const h = this.value ? THUMB_H + 4 : 16;
      ctx.beginPath();
      ctx.roundRect(margin, y + 2, w_, h, 6);
      ctx.fill();
      ctx.stroke();

      if (this.value) {
        const img = thumbFor(this.value, drawNode);
        const ih = THUMB_H - 4;
        if (img) {
          const iw = Math.min(ih * (img.naturalWidth / img.naturalHeight), 80);
          ctx.save();
          ctx.beginPath();
          ctx.roundRect(margin + 4, y + 6, iw, ih, 4);
          ctx.clip();
          ctx.drawImage(img, margin + 4, y + 6, iw, ih);
          ctx.restore();
          this.__thumbW = iw;
        } else {
          ctx.fillStyle = "#666";
          ctx.fillRect(margin + 4, y + 6, 40, ih);
          this.__thumbW = 40;
        }
        ctx.fillStyle = "#ddd";
        ctx.font = "10px Arial";
        const short = this.value.length > 22 ? "…" + this.value.slice(-21) : this.value;
        ctx.fillText(`ref: ${short}`, margin + this.__thumbW + 10, y + 6 + ih / 2 - 2);
        ctx.fillStyle = "#999";
        ctx.font = "9px Arial";
        ctx.fillText("(click to replace)", margin + this.__thumbW + 10, y + 6 + ih / 2 + 10);
        ctx.fillStyle = "#c66";
        ctx.font = "bold 11px Arial";
        ctx.fillText("✕", margin + w_ - 14, y + 14);
      } else {
        ctx.fillStyle = "#bbb";
        ctx.font = "10px Arial";
        ctx.textAlign = "center";
        ctx.fillText(`📷 load ref image for region ${idx + 1} (optional)`, margin + w_ / 2, y + 12);
        ctx.textAlign = "left";
      }
      ctx.restore();
    },
    mouse(event, pos, mNode) {
      const isDown = event.type === "pointerdown" || event.type === "mousedown";
      if (!isDown) return false;
      if (this.value && pos[0] > mNode.size[0] - 38) {
        const r = readRegions(mNode);
        if (r[idx]) {
          r[idx].ref_image = "";
          writeRegions(mNode, r);
          rebuildRows(mNode);
        }
        return true;
      }
      pickAndUploadRef(mNode, idx);
      return true;
    },
  };
  markTransient(w);
  return w;
}

// ---------------------------------------------------------------------------
// per-region control rows (enable / lora / strength / ref / remove)
// ---------------------------------------------------------------------------
function rebuildRows(node) {
  if (node.widgets) {
    node.widgets = node.widgets.filter((w) => !w.__rc_row);
  }
  const regions = readRegions(node);

  regions.forEach((region, idx) => {
    const enableW = node.addWidget(
      "toggle", `region ${idx + 1} enabled`, region.enable !== false,
      (v) => { const r = readRegions(node); if (r[idx]) { r[idx].enable = v; writeRegions(node, r); } node.setDirtyCanvas(true, true); },
      { on: "on", off: "off" }
    );
    markTransient(enableW);

    const loraW = node.addWidget(
      "combo", `region ${idx + 1} lora`, region.lora || "None",
      (v) => { const r = readRegions(node); if (r[idx]) { r[idx].lora = v; writeRegions(node, r); } node.setDirtyCanvas(true, true); },
      { values: LORA_LIST }
    );
    markTransient(loraW);

    const strW = node.addWidget(
      "number", `region ${idx + 1} strength`,
      typeof region.strength === "number" ? region.strength : 1.2,
      (v) => { const r = readRegions(node); if (r[idx]) { r[idx].strength = v; writeRegions(node, r); } },
      { min: -10.0, max: 10.0, step: 0.1, precision: 2 }
    );
    markTransient(strW);

    if (node.addCustomWidget) node.addCustomWidget(makeRefWidget(node, idx, region));
    else node.widgets.push(makeRefWidget(node, idx, region));

    const rmW = node.addWidget("button", `  ✕ remove region ${idx + 1}`, null, () => {
      const r = readRegions(node);
      r.splice(idx, 1);
      writeRegions(node, r);
      rebuildRows(node);
      node.setDirtyCanvas(true, true);
    });
    markTransient(rmW);
  });

  const sz = node.computeSize();
  node.size[1] = Math.max(node.size[1], sz[1]);
  node.setDirtyCanvas(true, true);
}

// ---------------------------------------------------------------------------
// box-drawing canvas (generalized from regional_character_lora's 2-box editor)
// ---------------------------------------------------------------------------
function buildCanvasWidget(node) {
  const canvas = document.createElement("canvas");
  canvas.style.width = "100%";
  canvas.style.display = "block";
  canvas.style.marginTop = "4px";
  canvas.style.borderRadius = "6px";
  canvas.style.touchAction = "none";
  canvas.style.cursor = "crosshair";

  let bgImage = null; // last-generated-output reference, purely visual, never serialized
  const CLEARBTN = 18;

  function draw() {
    const dpr = window.devicePixelRatio || 1;
    const cw = canvas.clientWidth || 260;
    const chh = canvas.clientHeight || 220;
    canvas.width = Math.round(cw * dpr);
    canvas.height = Math.round(chh * dpr);
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cw, chh);
    ctx.fillStyle = "#15151a";
    ctx.fillRect(0, 0, cw, chh);

    if (bgImage) {
      // "contain" fit: show the WHOLE image, letterboxed if needed -- never
      // crop, since faces can sit anywhere in the frame.
      const ir = bgImage.width / bgImage.height, cr = cw / chh;
      let dw, dh, dx, dy;
      if (ir > cr) { dw = cw; dh = dw / ir; dx = 0; dy = (chh - dh) / 2; }
      else { dh = chh; dw = dh * ir; dx = (cw - dw) / 2; dy = 0; }
      ctx.drawImage(bgImage, dx, dy, dw, dh);
      ctx.fillStyle = "rgba(0,0,0,0.2)";
      ctx.fillRect(dx, dy, dw, dh);
    }

    ctx.strokeStyle = "#3a3a42";
    ctx.strokeRect(0.5, 0.5, cw - 1, chh - 1);

    const regions = readRegions(node);
    const splitModeW = node.widgets?.find((w) => w.name === "split_mode");
    const active = !splitModeW || splitModeW.value === "manual";

    regions.forEach((reg, i) => {
      const col = hueColor(i, regions.length);
      const x = (reg.x ?? 0) * cw, y = (reg.y ?? 0) * chh;
      const w = (reg.w ?? 0.3) * cw, h = (reg.h ?? 0.3) * chh;
      ctx.globalAlpha = active ? (reg.enable !== false ? 1 : 0.35) : 0.25;
      ctx.fillStyle = hueColor(i, regions.length, bgImage ? 0.08 : 0.15);
      ctx.fillRect(x, y, w, h);
      ctx.lineWidth = 2;
      ctx.strokeStyle = col;
      ctx.strokeRect(x, y, w, h);
      ctx.fillStyle = col;
      ctx.fillRect(x + w - HANDLE, y + h - HANDLE, HANDLE, HANDLE);
      ctx.font = "11px sans-serif";
      ctx.textBaseline = "top";
      ctx.fillStyle = "#111";
      const label = `${i + 1} ${shortName(reg.lora)}`;
      ctx.fillText(label, x + 5, y + 4);
      ctx.globalAlpha = 1;
    });

    if (!active) {
      ctx.fillStyle = "#ddd";
      ctx.font = "11px sans-serif";
      ctx.fillText("set split_mode = manual to use these boxes", 6, chh - 16);
    }
    if (!regions.length) {
      ctx.fillStyle = "#888";
      ctx.font = "11px sans-serif";
      ctx.fillText("click '+ Add Region' below to start drawing boxes", 6, chh - 6);
    }
    if (bgImage) {
      ctx.fillStyle = "#000a";
      ctx.fillRect(cw - CLEARBTN - 6, 6, CLEARBTN, CLEARBTN);
      ctx.strokeStyle = "#aaa"; ctx.lineWidth = 1.5;
      const cx0 = cw - CLEARBTN - 6 + 5, cy0 = 6 + 5, cx1 = cw - 6 - 5, cy1 = 6 + CLEARBTN - 5;
      ctx.beginPath(); ctx.moveTo(cx0, cy0); ctx.lineTo(cx1, cy1);
      ctx.moveTo(cx1, cy0); ctx.lineTo(cx0, cy1); ctx.stroke();
    } else {
      ctx.fillStyle = "#666";
      ctx.font = "10px sans-serif";
      ctx.fillText("drop an image here, or load the latest output, to line up faces", 6, chh - 6);
    }
  }

  function onImageLoaded(img) {
    bgImage = img;
    draw();
    node.setDirtyCanvas(true, true);
  }

  const container = document.createElement("div");
  container.style.display = "flex";
  container.style.flexDirection = "column";
  container.appendChild(canvas);

  const toolbar = document.createElement("div");
  toolbar.style.cssText = "display:flex;align-items:center;gap:8px;margin-top:4px;";

  const loadBtn = document.createElement("button");
  loadBtn.textContent = "↻ Load latest output";
  loadBtn.style.cssText = "font-size:11px;padding:3px 8px;cursor:pointer;background:#2a2a32;"
    + "color:#ddd;border:1px solid #444;border-radius:4px;";
  loadBtn.onclick = async (e) => {
    e.stopPropagation();
    loadBtn.textContent = "…";
    const url = await fetchLatestOutputImageURL();
    loadBtn.textContent = "↻ Load latest output";
    if (!url) return;
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => onImageLoaded(img);
    img.src = url;
  };

  const autoLabel = document.createElement("label");
  autoLabel.style.cssText = "font-size:11px;color:#999;display:flex;align-items:center;gap:4px;cursor:pointer;";
  const autoCheckbox = document.createElement("input");
  autoCheckbox.type = "checkbox";
  autoLabel.appendChild(autoCheckbox);
  autoLabel.appendChild(document.createTextNode("auto after each run"));

  toolbar.appendChild(loadBtn);
  toolbar.appendChild(autoLabel);
  container.appendChild(toolbar);

  let executedHandler = null;
  autoCheckbox.addEventListener("change", () => {
    if (autoCheckbox.checked) {
      executedHandler = (e) => {
        const url = imageOutputURLFromExecutedEvent(e.detail);
        if (!url) return;
        const img = new Image();
        img.crossOrigin = "anonymous";
        img.onload = () => onImageLoaded(img);
        img.src = url;
      };
      api.addEventListener("executed", executedHandler);
    } else if (executedHandler) {
      api.removeEventListener("executed", executedHandler);
      executedHandler = null;
    }
  });
  const oldRemoved = node.onRemoved;
  node.onRemoved = function () {
    if (executedHandler) api.removeEventListener("executed", executedHandler);
    oldRemoved && oldRemoved.apply(this, arguments);
  };

  const widget = node.addDOMWidget("region_canvas", "rc_canvas", container, {
    getValue() { return ""; },
    setValue() {},
    getMinHeight() {
      const w = node.size ? node.size[0] - 20 : 220;
      return Math.round(Math.max(140, Math.min(w * 1.1, 380))) + 26;
    },
    hideOnZoom: false,
  });
  widget.serialize = false;
  widget.__rc_canvas = true;

  // -- interaction: move / resize boxes, click-to-clear-bg, drag-drop image --
  let drag = null;
  const toNorm = (e) => {
    const r = canvas.getBoundingClientRect();
    return [clamp01((e.clientX - r.left) / r.width), clamp01((e.clientY - r.top) / r.height)];
  };
  const onDown = (e) => {
    const r = canvas.getBoundingClientRect();
    const [nx, ny] = toNorm(e);

    if (bgImage) {
      const px = nx * r.width, py = ny * r.height;
      const bx0 = r.width - CLEARBTN - 6, by0 = 6;
      if (px >= bx0 && px <= bx0 + CLEARBTN && py >= by0 && py <= by0 + CLEARBTN) {
        bgImage = null; draw();
        return;
      }
    }

    const regions = readRegions(node);
    for (let i = regions.length - 1; i >= 0; i--) {
      const reg = regions[i];
      const rx = reg.x ?? 0, ry = reg.y ?? 0, rw = reg.w ?? 0.3, rh = reg.h ?? 0.3;
      if (Math.abs(nx - (rx + rw)) * r.width <= HANDLE && Math.abs(ny - (ry + rh)) * r.height <= HANDLE) {
        drag = { i, mode: "resize" }; break;
      }
      if (nx >= rx && nx <= rx + rw && ny >= ry && ny <= ry + rh) {
        drag = { i, mode: "move", ox: nx - rx, oy: ny - ry }; break;
      }
    }
    if (drag) {
      e.preventDefault();
      window.addEventListener("pointermove", onMove);
      window.addEventListener("pointerup", onUp);
    }
  };
  const onMove = (e) => {
    if (!drag) return;
    const [nx, ny] = toNorm(e);
    const regions = readRegions(node);
    const reg = regions[drag.i];
    if (!reg) return;
    if (drag.mode === "move") {
      const rw = reg.w ?? 0.3, rh = reg.h ?? 0.3;
      reg.x = clamp01(nx - drag.ox);
      reg.y = clamp01(ny - drag.oy);
      if (reg.x + rw > 1) reg.x = 1 - rw;
      if (reg.y + rh > 1) reg.y = 1 - rh;
    } else {
      reg.w = Math.max(MINSIZE, clamp01(nx - (reg.x ?? 0)));
      reg.h = Math.max(MINSIZE, clamp01(ny - (reg.y ?? 0)));
    }
    writeRegions(node, regions);
    draw();
  };
  const onUp = () => {
    drag = null;
    window.removeEventListener("pointermove", onMove);
    window.removeEventListener("pointerup", onUp);
  };
  canvas.addEventListener("pointerdown", onDown);

  canvas.addEventListener("dragover", (e) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = "copy";
  });
  canvas.addEventListener("drop", (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files && e.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) {
      const reader = new FileReader();
      reader.onload = () => {
        const img = new Image();
        img.onload = () => onImageLoaded(img);
        img.src = reader.result;
      };
      reader.readAsDataURL(file);
      return;
    }
    const url = e.dataTransfer.getData("text/uri-list") || e.dataTransfer.getData("text/plain");
    if (url) {
      const img = new Image();
      img.crossOrigin = "anonymous";
      img.onload = () => onImageLoaded(img);
      img.src = url;
    }
  });

  try { new ResizeObserver(() => draw()).observe(canvas); } catch (e) {}
  setTimeout(draw, 50);
  node.__rc_draw = draw;

  const oldResize = node.onResize;
  node.onResize = function () { oldResize && oldResize.apply(this, arguments); draw(); };

  return widget;
}

// ---------------------------------------------------------------------------
// extension registration
// ---------------------------------------------------------------------------
app.registerExtension({
  name: "RegioCraft.editor",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_TYPE) return;
    await ensureLoraList();

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
      const node = this;

      // seed the hidden regions_json widget if it's empty/default
      const jw = node.widgets?.find((w) => w.name === JSON_WIDGET);
      if (jw) {
        try {
          const parsed = JSON.parse(jw.value || "[]");
          if (!Array.isArray(parsed) || !parsed.length) jw.value = JSON.stringify(defaultRegions());
        } catch (e) {
          jw.value = JSON.stringify(defaultRegions());
        }
      }

      buildCanvasWidget(node);
      rebuildRows(node);

      node.addWidget("button", "+ Add Region", null, () => {
        const regions = readRegions(node);
        regions.push(defaultRegion(regions.length, regions.length + 1));
        writeRegions(node, regions);
        rebuildRows(node);
        node.__rc_draw && node.__rc_draw();
      });

      return r;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (o) {
      const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
      setTimeout(() => {
        rebuildRows(this);
        this.__rc_draw && this.__rc_draw();
      }, 0);
      return r;
    };

    const onDrawForeground = nodeType.prototype.onDrawForeground;
    nodeType.prototype.onDrawForeground = function (ctx) {
      if (onDrawForeground) onDrawForeground.apply(this, arguments);
    };
  },
});
