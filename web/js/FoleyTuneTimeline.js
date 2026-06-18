import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const PALETTE = {
    red: "#e74c3c", blue: "#3498db", green: "#2ecc71",
    yellow: "#f1c40f", purple: "#9b59b6", orange: "#e67e22",
};

function resolveColor(entry) {
    if (!entry) return PALETTE.blue;
    return PALETTE[entry.color] || PALETTE.blue;
}

const RULER_H = 22;
const THUMB_H = 60;
const TRACK_H = 48;
const PREVIEW_H = 220;    // frame-preview area below the timeline (scrub player)
const HANDLE_W = 6;       // visual width of the resize-handle indicator
const EDGE_GRAB = 10;     // hit tolerance around a segment edge (px, straddles it)
const CANVAS_H = RULER_H + TRACK_H;
const TOTAL_H = THUMB_H + CANVAS_H + PREVIEW_H;
const MIN_SEG_FRAMES = 1;   // shortest segment, in frames
const DEFAULT_FPS = 30;     // fallback when the video fps is unknown
const SNAP_SEC = 8;         // magnetic grid for segment boundaries (hold Shift to disable)
const SNAP_PX = 9;          // magnet range in px — beyond this, boundaries move freely
const SAFA_OVERLAP_SEC = 1.5;  // overlap pulled in at a SaFa seam (must match utils.py)
const PLAYHEAD_COLOR = "#ffcc00";

class TimelineEditor {
    constructor(node, container) {
        this.node = node;
        this.container = container;

        this.duration = 0;
        this.entries = [];
        this.segments = [];
        this.selectedIdx = -1;
        this.hoverIdx = -1;
        this.hoverEdge = null; // "left" | "right" | null
        this.drag = null;
        this._commitTimer = null;
        this.playhead = 0;       // scrub position in seconds (frame-snapped)
        this.videoPath = "";
        this.fps = DEFAULT_FPS;
        this._frameTimer = null;

        // Find and hide the segments_json widget. It's a multiline STRING, so it
        // owns a DOM <textarea>; flipping .type does NOT remove that element —
        // without hiding it explicitly it stays visible and inflates node height.
        this.segWidget = node.widgets?.find(w => w.name === "segments_json");
        if (this.segWidget) {
            this.segWidget.computeSize = () => [0, -4];
            this.segWidget.type = "converted-widget";
            // Hide the DOM <textarea> too (keeps the value serialized, unlike
            // .hidden which can drop it from widgets_values).
            const el = this.segWidget.inputEl
                || this.segWidget.element
                || (this.segWidget.options && this.segWidget.options.element);
            if (el && el.style) el.style.display = "none";
        }

        this._buildDOM();
        this._bindEvents();

        this._resizeObserver = new ResizeObserver(() => this._onResize());
        this._resizeObserver.observe(container);
    }

    _buildDOM() {
        // Thumbnail strip
        this.thumbStrip = document.createElement("div");
        this.thumbStrip.style.cssText =
            `width:100%;height:${THUMB_H}px;background:#111;overflow:hidden;position:relative;`;
        this.thumbImg = document.createElement("img");
        // Stretch the full-duration sprite across the strip so thumbnail x maps
        // linearly to time (aligns with the ruler below).
        this.thumbImg.style.cssText =
            "width:100%;height:100%;position:absolute;left:0;top:0;object-fit:fill;";
        this.thumbStrip.appendChild(this.thumbImg);
        this.container.appendChild(this.thumbStrip);

        // Canvas for ruler + segments
        this.canvas = document.createElement("canvas");
        this.canvas.style.cssText = `width:100%;height:${CANVAS_H}px;display:block;`;
        this.container.appendChild(this.canvas);
        this.ctx = this.canvas.getContext("2d");

        // Scrub player: a larger frame at the playhead, for precise alignment.
        // Click/drag the ruler to move the playhead.
        this.preview = document.createElement("div");
        this.preview.style.cssText =
            `width:100%;height:${PREVIEW_H}px;background:#0a0a12;position:relative;` +
            `display:flex;align-items:center;justify-content:center;overflow:hidden;`;
        this.previewImg = document.createElement("img");
        this.previewImg.style.cssText =
            "max-width:100%;max-height:100%;object-fit:contain;display:block;";
        this.preview.appendChild(this.previewImg);
        this.previewLabel = document.createElement("div");
        this.previewLabel.style.cssText =
            "position:absolute;left:6px;top:4px;font:11px monospace;color:#ffcc00;" +
            "background:rgba(0,0,0,0.55);padding:1px 5px;border-radius:3px;pointer-events:none;";
        this.previewLabel.textContent = "scrub the ruler ▸";
        this.preview.appendChild(this.previewLabel);
        this.container.appendChild(this.preview);
    }

    _bindEvents() {
        const c = this.canvas;
        c.addEventListener("pointerdown", (e) => this._onPointerDown(e));
        c.addEventListener("pointermove", (e) => this._onPointerMove(e));
        c.addEventListener("pointerup", (e) => this._onPointerUp(e));
        c.addEventListener("pointerleave", () => this._onPointerLeave());
        c.addEventListener("dblclick", (e) => this._onDblClick(e));
        c.addEventListener("contextmenu", (e) => this._onContextMenu(e));
    }

    // --- Coordinate helpers ---

    _dpr() { return window.devicePixelRatio || 1; }

    _canvasW() { return this.canvas.clientWidth; }

    _secToX(sec) {
        return this.duration > 0 ? (sec / this.duration) * this._canvasW() : 0;
    }

    _xToSec(x) {
        // Snap to the nearest frame boundary.
        return this._snapSec(this._xToSecRaw(x));
    }

    _xToSecRaw(x) {
        return this.duration > 0 ? Math.max(0, Math.min(this.duration, (x / this._canvasW()) * this.duration)) : 0;
    }

    // --- Frame helpers (the timeline works in frames at the video fps) ---

    _frame(sec) { return Math.round(sec * this.fps); }       // sec -> frame index
    _snapSec(sec) { return this._frame(sec) / this.fps; }    // snap sec to a frame
    _minSeg() { return MIN_SEG_FRAMES / this.fps; }          // shortest segment, sec

    // MAGNETIC snap toward the SNAP_SEC grid (e.g. 8s) and the clip ends. Only
    // pulls when the cursor is within SNAP_PX of a line — otherwise the boundary
    // moves freely (frame-snapped), so any position/size is reachable. Hold
    // Shift (fine=true) to disable the magnet entirely.
    _snapSegSec(sec, fine) {
        if (fine) return this._snapSec(sec);
        let best = null, bestD = Infinity;
        for (const c of [Math.round(sec / SNAP_SEC) * SNAP_SEC, 0, this.duration]) {
            const d = Math.abs(sec - c);
            if (d < bestD) { bestD = d; best = c; }
        }
        const pxPerSec = this._canvasW() / Math.max(1e-6, this.duration);
        if (best !== null && bestD * pxPerSec <= SNAP_PX) {
            return this._snapSec(Math.max(0, Math.min(this.duration, best)));
        }
        return this._snapSec(sec);  // outside the magnet zone: free placement
    }

    // Nearest segment to the right/left of `seg` (by facing edge). Used so a
    // dragged edge pushes its neighbour instead of overlapping it.
    _neighborRight(seg) {
        let best = null;
        for (const o of this.segments) {
            if (o === seg || o.start_sec < seg.start_sec) continue;
            if (!best || o.start_sec < best.start_sec) best = o;
        }
        return best;
    }
    _neighborLeft(seg) {
        let best = null;
        for (const o of this.segments) {
            if (o === seg || o.end_sec > seg.end_sec) continue;
            if (!best || o.end_sec > best.end_sec) best = o;
        }
        return best;
    }

    _pointerPos(e) {
        // getBoundingClientRect reflects LiteGraph's zoom (CSS transform), but
        // _secToX/_canvasW use the canvas's logical width. Rescale screen px to
        // logical px so hit-testing (esp. the ~10px edge zone) is correct at any
        // zoom — otherwise edges become ungrabbable when zoomed in/out.
        const r = this.canvas.getBoundingClientRect();
        const sx = r.width ? this.canvas.clientWidth / r.width : 1;
        const sy = r.height ? this.canvas.clientHeight / r.height : 1;
        return { x: (e.clientX - r.left) * sx, y: (e.clientY - r.top) * sy };
    }

    _hitTest(x, y) {
        if (y < RULER_H || y > RULER_H + TRACK_H) return { idx: -1, edge: null };
        // First pass: edges win over bodies, and the grab zone straddles the
        // boundary (a few px outside the segment counts too) so handles are
        // actually grabbable. Reverse order = topmost segment first.
        for (let i = this.segments.length - 1; i >= 0; i--) {
            const seg = this.segments[i];
            const left = this._secToX(seg.start_sec);
            const right = this._secToX(seg.end_sec);
            const nearLeft = Math.abs(x - left) <= EDGE_GRAB;
            const nearRight = Math.abs(x - right) <= EDGE_GRAB;
            if (nearLeft && nearRight) {
                // Tiny segment — pick the closer edge.
                return { idx: i, edge: (x - left <= right - x) ? "left" : "right" };
            }
            if (nearLeft) return { idx: i, edge: "left" };
            if (nearRight) return { idx: i, edge: "right" };
        }
        // Second pass: clicking the body = move.
        for (let i = this.segments.length - 1; i >= 0; i--) {
            const seg = this.segments[i];
            const left = this._secToX(seg.start_sec);
            const right = this._secToX(seg.end_sec);
            if (x >= left && x <= right) return { idx: i, edge: null };
        }
        return { idx: -1, edge: null };
    }

    // --- Events ---

    _onPointerDown(e) {
        // Only the left button starts a drag/selection. A right-button down
        // would set this.drag, then contextmenu deletes the segment, leaving
        // this.drag pointing at a removed index (crash on next move).
        if (e.button !== 0) return;
        e.stopPropagation();
        const { x, y } = this._pointerPos(e);

        // Ruler band = scrub the playhead (frame preview below).
        if (y < RULER_H) {
            this.drag = { type: "scrub" };
            this.canvas.setPointerCapture(e.pointerId);
            this._setPlayhead(this._xToSecRaw(x));
            return;
        }

        // SaFa seam badge? Toggle it (blend this zone's left seam) instead of dragging.
        for (const b of (this._seamBadges || [])) {
            if ((x - b.x) ** 2 + (y - b.y) ** 2 <= (b.r + 2) ** 2) {
                this._toggleSeam(b.ref);
                return;
            }
        }

        const hit = this._hitTest(x, y);

        if (hit.idx < 0) {
            this.selectedIdx = -1;
            this.render();
            return;
        }

        this.selectedIdx = hit.idx;
        const seg = this.segments[hit.idx];

        if (hit.edge) {
            this.drag = {
                type: `resize-${hit.edge}`,
                idx: hit.idx,
                startX: x,
                origStart: seg.start_sec,
                origEnd: seg.end_sec,
            };
        } else {
            this.drag = {
                type: "move",
                idx: hit.idx,
                startX: x,
                origStart: seg.start_sec,
                origEnd: seg.end_sec,
            };
        }
        this.canvas.setPointerCapture(e.pointerId);
        this.render();
    }

    _onPointerMove(e) {
        e.stopPropagation();
        const { x, y } = this._pointerPos(e);

        if (this.drag) {
            if (this.drag.type === "scrub") {
                this._setPlayhead(this._xToSecRaw(x));
                return;
            }
            const seg = this.segments[this.drag.idx];
            if (!seg) { this.drag = null; return; }  // segment removed mid-drag

            if (this.drag.type === "move") {
                // Raw (unsnapped) pixel delta keeps dragging smooth; snap the
                // final position once.
                const dSecRaw = this._canvasW() > 0
                    ? ((x - this.drag.startX) / this._canvasW()) * this.duration : 0;
                const len = this.drag.origEnd - this.drag.origStart;
                let ns = this._snapSegSec(Math.max(0, Math.min(this.duration - len, this.drag.origStart + dSecRaw)), e.shiftKey);
                // Block overlap: keep the segment inside the room between its
                // neighbours (pushes nothing — a move just can't ride over them).
                let lo = 0, hi = this.duration - len;
                for (const o of this.segments) {
                    if (o === seg) continue;
                    if (o.end_sec <= this.drag.origStart + 1e-6) lo = Math.max(lo, o.end_sec);
                    else if (o.start_sec >= this.drag.origEnd - 1e-6) hi = Math.min(hi, o.start_sec - len);
                }
                if (hi < lo) hi = lo;
                ns = Math.max(lo, Math.min(hi, ns));
                seg.start_sec = ns;
                seg.end_sec = ns + len;
            } else if (this.drag.type === "resize-left") {
                // Edge snaps to the SNAP_SEC grid (Shift = per-frame), clamped to [0,dur].
                let ns = Math.min(seg.end_sec - this._minSeg(), this._snapSegSec(this._xToSecRaw(x), e.shiftKey));
                // Push (or roll, when already touching) the left neighbour's
                // right edge instead of overlapping it.
                const nb = this._neighborLeft(seg);
                if (nb && (ns < nb.end_sec || Math.abs(seg.start_sec - nb.end_sec) < 1e-6)) {
                    ns = Math.max(ns, nb.start_sec + this._minSeg());  // don't crush it
                    nb.end_sec = ns;
                }
                seg.start_sec = ns;
            } else if (this.drag.type === "resize-right") {
                let ns = Math.max(seg.start_sec + this._minSeg(), this._snapSegSec(this._xToSecRaw(x), e.shiftKey));
                const nb = this._neighborRight(seg);
                if (nb && (ns > nb.start_sec || Math.abs(seg.end_sec - nb.start_sec) < 1e-6)) {
                    ns = Math.min(ns, nb.end_sec - this._minSeg());
                    nb.start_sec = ns;
                }
                seg.end_sec = ns;
            }
            this._commitDebounced();
            this.render();
            return;
        }

        // Hover feedback
        const hit = this._hitTest(x, y);
        const changed = hit.idx !== this.hoverIdx || hit.edge !== this.hoverEdge;
        this.hoverIdx = hit.idx;
        this.hoverEdge = hit.edge;

        if (y < RULER_H) {
            this.canvas.style.cursor = "ew-resize";  // ruler = scrub zone
        } else if (hit.edge) {
            this.canvas.style.cursor = "ew-resize";
        } else if (hit.idx >= 0) {
            this.canvas.style.cursor = "grab";
        } else {
            this.canvas.style.cursor = "default";
        }

        if (changed) this.render();
    }

    _onPointerUp(e) {
        if (this.drag) {
            const wasScrub = this.drag.type === "scrub";
            this.drag = null;
            this.canvas.releasePointerCapture(e.pointerId);
            if (!wasScrub) this._commitFlush();  // scrub doesn't change segments
            this.render();
        }
    }

    // --- Scrub player ---

    _setPlayhead(sec) {
        // Snap the playhead to a frame boundary too.
        this.playhead = this._snapSec(Math.max(0, Math.min(this.duration || 0, sec)));
        this.previewLabel.textContent = `frame ${this._frame(this.playhead)}  ·  ${this.playhead.toFixed(2)}s`;
        this._loadPreviewFrame();
        this.render();
    }

    _loadPreviewFrame() {
        if (!this.videoPath) return;
        // Debounce so dragging the playhead doesn't spawn an ffmpeg per pixel.
        if (this._frameTimer) clearTimeout(this._frameTimer);
        this._frameTimer = setTimeout(() => {
            this._frameTimer = null;
            this.previewImg.src = api.apiURL(
                "/foleytune/timeline_frame?video_path=" + encodeURIComponent(this.videoPath) +
                "&t=" + this.playhead.toFixed(2));
        }, 80);
    }

    _onPointerLeave() {
        if (!this.drag) {
            this.hoverIdx = -1;
            this.hoverEdge = null;
            this.canvas.style.cursor = "default";
            this.render();
        }
    }

    // Prompt for a LoRA entry index (skips the prompt for a single entry).
    // Returns the index, or null if cancelled/invalid.
    _pickEntry(defaultIdx) {
        if (this.entries.length <= 1) return 0;
        const labels = this.entries.map((en, i) => `${i}: ${en.label}`).join("\n");
        const input = prompt(`Select LoRA entry:\n${labels}`, String(defaultIdx ?? 0));
        if (input === null) return null;
        const i = parseInt(input, 10);
        return (i >= 0 && i < this.entries.length) ? i : null;
    }

    _entryIndexOf(seg) {
        if (seg.entry_id != null) {
            const i = this.entries.findIndex(en => en && en.id === seg.entry_id);
            if (i >= 0) return i;
        }
        return seg.entry_index ?? 0;
    }

    _assignEntry(seg, idx) {
        seg.entry_id = this.entries[idx].id;
        seg.entry_index = idx;
        seg.strength = this.entries[idx].strength;
    }

    _onDblClick(e) {
        e.stopPropagation();
        e.preventDefault();
        if (!this.entries.length || this.duration <= 0) return;

        const { x, y } = this._pointerPos(e);

        // Double-click on a zone = SWITCH its LoRA; on empty space = ADD a zone.
        const hit = this._hitTest(x, y);
        if (hit.idx >= 0) {
            const seg = this.segments[hit.idx];
            const idx = this._pickEntry(this._entryIndexOf(seg));
            if (idx === null) return;
            this._assignEntry(seg, idx);
            this._commitFlush();
            this.render();
            return;
        }

        const idx = this._pickEntry(0);
        if (idx === null) return;
        const sec = this._xToSec(x);
        const startSec = this._snapSec(Math.max(0, sec - 2));
        const endSec = this._snapSec(Math.min(this.duration, sec + 2));
        const seg = { start_sec: startSec, end_sec: endSec };
        this._assignEntry(seg, idx);
        this.segments.push(seg);
        this.segments.sort((a, b) => a.start_sec - b.start_sec);
        this.selectedIdx = this.segments.length - 1;
        this._commitFlush();
        this.render();
    }

    // Adjustable SaFa seam overlap (seconds), from the safa_overlap widget;
    // falls back to the default constant. Used by the seam toggle + auto-populate.
    _safaOverlap() {
        const wgt = this.node?.widgets?.find(w => w.name === "safa_overlap");
        const v = wgt ? Number(wgt.value) : SAFA_OVERLAP_SEC;
        return (Number.isFinite(v) && v > 0) ? v : SAFA_OVERLAP_SEC;
    }

    _autoSafa() {
        const wgt = this.node?.widgets?.find(w => w.name === "auto_populate_safa");
        return !!(wgt && wgt.value);
    }

    // Fill the timeline with 8s zones, assigning entries in chain order
    // (cycling). SaFa ON → overlapping 8s zones (mirrors compute_chunk_boundaries
    // / plain auto-chunked sampling); OFF → contiguous 8s tiles (hard cuts, last
    // zone is the remainder). Replaces any existing zones.
    _autoPopulate() {
        if (!(this.duration > 0) || !this.entries.length) return;
        const W = SNAP_SEC, dur = this.duration, safa = this._autoSafa();
        let chunks;
        if (dur <= W + 1e-6) {
            chunks = [[0, dur]];
        } else if (safa) {
            // Fixed safa_overlap per seam (same as the manual seam toggle) —
            // 8s zones striding by W-overlap, NOT an evened-out overlap. Last
            // zone is the remainder.
            const stride = W - this._safaOverlap();
            const n = Math.ceil((dur - W) / stride) + 1;
            chunks = [];
            for (let i = 0; i < n; i++) { const s = i * stride; chunks.push([s, Math.min(s + W, dur)]); }
        } else {
            const n = Math.ceil(dur / W);
            chunks = [];
            for (let i = 0; i < n; i++) chunks.push([i * W, Math.min((i + 1) * W, dur)]);
        }
        this.segments = chunks.map(([s, eSec], i) => {
            const seg = { start_sec: this._snapSec(s), end_sec: this._snapSec(eSec) };
            this._assignEntry(seg, i % this.entries.length);
            return seg;
        });
        this.selectedIdx = -1;
        this._commitFlush();
        this.render();
    }

    _onContextMenu(e) {
        e.preventDefault();
        e.stopPropagation();
        const { x, y } = this._pointerPos(e);
        const hit = this._hitTest(x, y);
        if (hit.idx < 0) { this._closeMenu(); return; }
        // Select the zone and open a menu — deletion is a deliberate choice now.
        this.selectedIdx = hit.idx;
        this.render();
        this._showContextMenu(e.clientX, e.clientY, hit.idx);
    }

    _closeMenu() {
        if (this._menu) { this._menu.remove(); this._menu = null; }
        if (this._menuClose) { window.removeEventListener("pointerdown", this._menuClose, true); this._menuClose = null; }
        if (this._menuKey) { window.removeEventListener("keydown", this._menuKey, true); this._menuKey = null; }
    }

    _showContextMenu(clientX, clientY, idx) {
        this._closeMenu();
        const seg = this.segments[idx];
        if (!seg) return;

        const menu = document.createElement("div");
        menu.style.cssText =
            "position:fixed;z-index:10000;background:#23272e;border:1px solid #454c59;" +
            "border-radius:5px;padding:4px 0;font:12px sans-serif;color:#dde;min-width:160px;" +
            "box-shadow:0 3px 12px rgba(0,0,0,0.55);user-select:none;";

        const addItem = (label, fn, opts = {}) => {
            const it = document.createElement("div");
            it.textContent = label;
            it.style.cssText = "padding:5px 14px;white-space:nowrap;" +
                (opts.header ? "color:#7e8696;font-size:11px;cursor:default;"
                             : "cursor:pointer;") +
                (opts.danger ? "color:#ff8b8b;" : "") +
                (opts.active ? "color:#fff;font-weight:bold;" : "");
            if (!opts.header) {
                it.addEventListener("mouseenter", () => { it.style.background = "#39414f"; });
                it.addEventListener("mouseleave", () => { it.style.background = ""; });
                it.addEventListener("click", (ev) => { ev.stopPropagation(); this._closeMenu(); fn(); });
            }
            menu.appendChild(it);
        };
        const sep = () => {
            const s = document.createElement("div");
            s.style.cssText = "height:1px;background:#454c59;margin:4px 0;";
            menu.appendChild(s);
        };

        if (this.entries.length > 1) {
            addItem("Switch LoRA", null, { header: true });
            const cur = this._entryIndexOf(seg);
            this.entries.forEach((en, i) => {
                addItem((i === cur ? "● " : "    ") + (en.label || `entry ${i}`),
                        () => { this._assignEntry(seg, i); this._commitFlush(); this.render(); },
                        { active: i === cur });
            });
            sep();
        }
        addItem("Delete zone", () => {
            const at = this.segments.indexOf(seg);
            if (at >= 0) this.segments.splice(at, 1);
            this.selectedIdx = -1;
            this.drag = null;
            this._commitFlush();
            this.render();
        }, { danger: true });

        document.body.appendChild(menu);
        // Keep on-screen.
        const r = menu.getBoundingClientRect();
        const left = Math.min(clientX, window.innerWidth - r.width - 4);
        const top = Math.min(clientY, window.innerHeight - r.height - 4);
        menu.style.left = Math.max(4, left) + "px";
        menu.style.top = Math.max(4, top) + "px";
        this._menu = menu;

        // Dismiss on outside click / Escape (deferred so this event doesn't self-close it).
        this._menuClose = (ev) => { if (!menu.contains(ev.target)) this._closeMenu(); };
        this._menuKey = (ev) => { if (ev.key === "Escape") this._closeMenu(); };
        setTimeout(() => {
            window.addEventListener("pointerdown", this._menuClose, true);
            window.addEventListener("keydown", this._menuKey, true);
        }, 0);
    }

    // --- State sync ---

    _commitDebounced() {
        if (this._commitTimer) clearTimeout(this._commitTimer);
        this._commitTimer = setTimeout(() => {
            this._commitTimer = null;
            this._commitFlush();
        }, 120);
    }

    _commitFlush() {
        if (this._commitTimer) {
            clearTimeout(this._commitTimer);
            this._commitTimer = null;
        }
        const json = JSON.stringify(this.segments);
        // Backend reads the widget value (positional in widgets_values).
        if (this.segWidget) this.segWidget.value = json;
        // Mirror into node.properties, which serializes BY NAME — so the data
        // survives widget reordering / node-definition changes that would
        // otherwise scramble the positional widget slot and scrap the segments.
        this.node.properties = this.node.properties || {};
        this.node.properties.foleytune_segments = json;
    }

    // --- Resize ---

    _onResize() {
        this._enforceWidth();
        this._updateCanvasSize();
        this.render();
    }

    // A ComfyUI-wide bug collapses preview/DOM-widget elements to ~half width
    // when the node is selected/re-laid-out (VHS Load Video shows it too). Guard:
    // if our element is narrower than a sibling widget that stayed full width,
    // force it back. The _enforcing flag breaks the ResizeObserver feedback loop.
    _referenceWidth() {
        let w = 0;
        for (const widget of (this.node.widgets || [])) {
            const el = widget.inputEl || widget.element;
            if (el && el !== this.container && el.offsetWidth > w) w = el.offsetWidth;
        }
        if (!w && this.container.parentElement) w = this.container.parentElement.clientWidth;
        return w;
    }

    _enforceWidth() {
        if (this._enforcing) return;
        const ref = this._referenceWidth();
        if (ref > 0 && this.container.clientWidth < ref - 2) {
            this._enforcing = true;
            this.container.style.width = ref + "px";
            requestAnimationFrame(() => { this._enforcing = false; });
        }
    }

    _updateCanvasSize() {
        const dpr = this._dpr();
        const w = this._canvasW();
        this.canvas.width = Math.round(w * dpr);
        this.canvas.height = Math.round(CANVAS_H * dpr);
        this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        // Scale thumbnail to match
        this.thumbImg.style.width = this.thumbStrip.clientWidth + "px";
    }

    // --- Rendering ---

    render() {
        const ctx = this.ctx;
        const w = this._canvasW();
        if (w <= 0 || this.duration <= 0) return;

        // Keep the backing store synced to the current layout width. A LiteGraph
        // zoom changes the displayed width without firing the ResizeObserver, so
        // otherwise drawing coords (_secToX) drift vs the mouse, growing to the
        // right — most visible on the playhead.
        if (this.canvas.width !== Math.round(w * this._dpr())) {
            this._updateCanvasSize();
        }

        ctx.clearRect(0, 0, w, CANVAS_H);

        this._drawRuler(ctx, w);
        this._drawTrack(ctx, w);
        this._drawSegments(ctx, w);
        this._drawChunkPlan(ctx, w);
        this._drawSeamBadges(ctx);
        this._drawPlayhead(ctx, w);
    }

    // A seam is SaFa iff the two zones OVERLAP in time (the UI shifts a zone
    // left by SAFA_OVERLAP_SEC to create the overlap). Single source of truth =
    // the zone positions, read straight through by the sampler.
    _seamOverlap(prev, cur) { return prev.end_sec - cur.start_sec; }  // >0 = SaFa, ~0 = hard cut

    // Toggle the seam to the LEFT of `seg`: if it currently overlaps its
    // predecessor, push back to touching (hard cut); else pull left by
    // SAFA_OVERLAP_SEC to overlap (SaFa). Shifts this zone AND all later ones so
    // the rest of the chain rides along (you SEE the clip shift).
    _toggleSeam(seg) {
        const sorted = [...this.segments].sort((a, b) => a.start_sec - b.start_sec);
        const i = sorted.indexOf(seg);
        if (i <= 0) return;  // first zone has no left seam
        const prev = sorted[i - 1];
        const overlapping = this._seamOverlap(prev, seg) > 1e-3;
        const targetStart = overlapping ? prev.end_sec : prev.end_sec - this._safaOverlap();
        let delta = this._snapSec(targetStart) - seg.start_sec;
        if (delta < 0) delta = Math.max(delta, -seg.start_sec);  // don't push the chain below 0
        for (let j = i; j < sorted.length; j++) {
            sorted[j].start_sec += delta;
            sorted[j].end_sec += delta;
        }
        this._commitFlush();
        this.render();
    }

    // Overlay the chunk plan straight from the zone positions: ✕-hatch where
    // zones overlap (SaFa seam), a sharp dashed line where they touch (hard cut),
    // and a "cN" index per zone at its centre.
    _drawChunkPlan(ctx, w) {
        if (!(this.duration > 0) || !this.segments.length) return;
        const segs = [...this.segments].sort((a, b) => a.start_sec - b.start_sec);
        const yTop = RULER_H, yBot = RULER_H + TRACK_H;
        ctx.save();
        for (let i = 1; i < segs.length; i++) {
            const ov = this._seamOverlap(segs[i - 1], segs[i]);
            if (ov > 1e-3) {  // SaFa: shaded ✕-hatch over the overlap
                const x1 = this._secToX(segs[i].start_sec), x2 = this._secToX(segs[i - 1].end_sec);
                if (x2 - x1 < 1) continue;
                ctx.save();
                ctx.globalAlpha = 0.5;
                ctx.fillStyle = "rgba(120,170,255,0.55)";
                ctx.fillRect(x1, yTop + 3, x2 - x1, yBot - yTop - 6);
                ctx.strokeStyle = "#fff"; ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.moveTo(x1, yTop + 3); ctx.lineTo(x2, yBot - 3);
                ctx.moveTo(x1, yBot - 3); ctx.lineTo(x2, yTop + 3);
                ctx.stroke();
                ctx.restore();
            } else {  // hard cut: sharp dashed line at the touch
                const x = this._secToX(segs[i].start_sec);
                ctx.strokeStyle = "rgba(255,175,45,0.9)"; ctx.lineWidth = 1;
                ctx.setLineDash([3, 3]);
                ctx.beginPath(); ctx.moveTo(x, yTop); ctx.lineTo(x, yBot); ctx.stroke();
                ctx.setLineDash([]);
            }
        }
        ctx.fillStyle = "rgba(255,205,90,0.95)";
        ctx.font = "bold 9px monospace";
        ctx.textAlign = "center"; ctx.textBaseline = "top";
        for (let i = 0; i < segs.length; i++) {
            ctx.fillText(`c${i}`, this._secToX((segs[i].start_sec + segs[i].end_sec) / 2), yTop + 2);
        }
        ctx.restore();
    }

    // A clickable badge near each zone's left edge toggles SaFa on that seam.
    // Lit ≈ = overlapping (SaFa), dim | = touching (hard cut). Geometry-driven.
    _drawSeamBadges(ctx) {
        this._seamBadges = [];
        if (!(this.duration > 0)) return;
        const segs = [...this.segments].sort((a, b) => a.start_sec - b.start_sec);
        const y = RULER_H + 12, r = 7;
        for (let i = 1; i < segs.length; i++) {  // first zone has no left seam
            const s = segs[i];
            const left = this._secToX(s.start_sec), right = this._secToX(s.end_sec);
            if (right - left < 36) continue;  // too narrow for a badge
            const x = left + 14;               // inset past the resize-edge grab zone
            const on = this._seamOverlap(segs[i - 1], s) > 1e-3;
            ctx.save();
            ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2);
            ctx.fillStyle = on ? "rgba(90,150,255,0.95)" : "rgba(35,39,51,0.95)";
            ctx.fill();
            ctx.lineWidth = 1.5; ctx.strokeStyle = on ? "#d6e6ff" : "#7a7f8c";
            ctx.stroke();
            ctx.fillStyle = on ? "#fff" : "#aab";
            ctx.font = "bold 10px sans-serif"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
            ctx.fillText(on ? "≈" : "|", x, y + 0.5);
            ctx.restore();
            this._seamBadges.push({ x, y, r, ref: s });
        }
    }

    _drawPlayhead(ctx) {
        if (this.duration <= 0) return;
        const x = this._secToX(this.playhead);
        ctx.fillStyle = PLAYHEAD_COLOR;
        ctx.fillRect(x - 0.75, 0, 1.5, CANVAS_H);
        // Triangle handle at the top of the ruler.
        ctx.beginPath();
        ctx.moveTo(x - 5, 0);
        ctx.lineTo(x + 5, 0);
        ctx.lineTo(x, 7);
        ctx.closePath();
        ctx.fill();
    }

    _totalFrames() { return Math.max(1, Math.round(this.duration * this.fps)); }

    _tickStepFrames(w) {
        // Frame interval that keeps tick labels at least ~46px apart.
        const pxPerFrame = w / this._totalFrames();
        const rawStep = 46 / pxPerFrame;
        const nice = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 1200, 3000, 6000];
        return nice.find(s => s >= rawStep) || 6000;
    }

    _drawRuler(ctx, w) {
        ctx.fillStyle = "#1a1a2e";
        ctx.fillRect(0, 0, w, RULER_H);
        ctx.font = "10px monospace";

        const totalF = this._totalFrames();
        const stepF = this._tickStepFrames(w);

        for (let f = 0; f <= totalF; f += stepF) {
            const x = this._secToX(f / this.fps);
            const isMajor = (f % (stepF * 5)) === 0;

            ctx.fillStyle = "#555";
            ctx.fillRect(x, 0, 1, isMajor ? RULER_H - 4 : RULER_H - 10);

            if (isMajor || (w / totalF) * stepF > 50) {
                ctx.fillStyle = "#999";
                ctx.fillText(`${f}`, x + 3, RULER_H - 3);
            }
        }

        // Bottom border
        ctx.fillStyle = "#333";
        ctx.fillRect(0, RULER_H - 1, w, 1);
    }

    _drawTrack(ctx, w) {
        ctx.fillStyle = "#0e0e1a";
        ctx.fillRect(0, RULER_H, w, TRACK_H);

        // Grid lines at each ruler tick (frames) for alignment reference.
        ctx.fillStyle = "#1a1a2e";
        const totalF = this._totalFrames();
        const stepF = this._tickStepFrames(w);
        for (let f = 0; f <= totalF; f += stepF) {
            ctx.fillRect(this._secToX(f / this.fps), RULER_H, 1, TRACK_H);
        }

        // Brighter lines at each SNAP_SEC boundary — these are the snap targets.
        ctx.fillStyle = "#3a5a8c";
        for (let s = SNAP_SEC; s < this.duration; s += SNAP_SEC) {
            ctx.fillRect(this._secToX(s), RULER_H, 1, TRACK_H);
        }
    }

    _drawSegments(ctx, w) {
        const dragIdx = this.drag?.idx ?? -1;

        // Two-pass: non-dragged first, then dragged on top
        const order = this.segments.map((_, i) => i)
            .sort((a, b) => (a === dragIdx ? 1 : 0) - (b === dragIdx ? 1 : 0));

        for (const i of order) {
            this._drawSegment(ctx, i);
        }
    }

    _entryFor(seg) {
        // Resolve by stable id first (survives chain reorder), then by index.
        if (seg.entry_id != null) {
            const byId = this.entries.find(e => e && e.id === seg.entry_id);
            if (byId) return byId;
        }
        return this.entries[seg.entry_index] || {};
    }

    _drawSegment(ctx, i) {
        const seg = this.segments[i];
        const entry = this._entryFor(seg);
        const color = resolveColor(entry);
        // ✎ marks an entry with a per-segment prompt (see entry node).
        const hasPrompt = !!(entry.prompt && entry.prompt.trim());
        const label = (hasPrompt ? "✎ " : "") + (entry.label || "LoRA");

        const x = this._secToX(seg.start_sec);
        const x2 = this._secToX(seg.end_sec);
        const segW = x2 - x;
        const y = RULER_H + 3;
        const h = TRACK_H - 6;
        const r = 4;

        const isSelected = i === this.selectedIdx;
        const isHover = i === this.hoverIdx && !this.drag;
        const isDragging = this.drag?.idx === i;

        // Background fill
        ctx.save();
        ctx.globalAlpha = isDragging ? 0.9 : isHover ? 0.85 : 0.7;
        ctx.fillStyle = color;
        this._roundRect(ctx, x, y, segW, h, r);
        ctx.fill();
        ctx.restore();

        // Border
        ctx.save();
        ctx.strokeStyle = isDragging ? "#ffd700" : isSelected ? "#fff" : color;
        ctx.lineWidth = isSelected || isDragging ? 2 : 1;
        this._roundRect(ctx, x, y, segW, h, r);
        ctx.stroke();
        ctx.restore();

        // Label text
        if (segW > 20) {
            ctx.save();
            ctx.fillStyle = "#fff";
            ctx.font = "bold 11px sans-serif";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.shadowColor = "#000";
            ctx.shadowBlur = 3;

            const centerX = x + segW / 2;
            const centerY = y + h / 2;

            // Clip to segment bounds
            ctx.beginPath();
            this._roundRect(ctx, x + 2, y, segW - 4, h, r);
            ctx.clip();

            ctx.fillText(label, centerX, centerY - 6);

            // Frame range + duration (frames / fps, so it matches the cut exactly).
            ctx.font = "9px monospace";
            ctx.fillStyle = "#ccc";
            ctx.shadowBlur = 0;
            const sf = this._frame(seg.start_sec);
            const ef = this._frame(seg.end_sec);
            const durSec = (ef - sf) / this.fps;
            ctx.fillText(`f${sf}–f${ef} · ${durSec.toFixed(2)}s`, centerX, centerY + 8);

            ctx.restore();
        }

        // Resize handle indicators — full height + bright so the grab zone is
        // discoverable (the hit tolerance is EDGE_GRAB px around each edge).
        if (isHover || isSelected || isDragging) {
            ctx.fillStyle = "rgba(255,255,255,0.85)";
            ctx.fillRect(x, y, 3, h);
            ctx.fillRect(x2 - 3, y, 3, h);
        }
    }

    _roundRect(ctx, x, y, w, h, r) {
        ctx.beginPath();
        ctx.moveTo(x + r, y);
        ctx.lineTo(x + w - r, y);
        ctx.arcTo(x + w, y, x + w, y + r, r);
        ctx.lineTo(x + w, y + h - r);
        ctx.arcTo(x + w, y + h, x + w - r, y + h, r);
        ctx.lineTo(x + r, y + h);
        ctx.arcTo(x, y + h, x, y + h - r, r);
        ctx.lineTo(x, y + r);
        ctx.arcTo(x, y, x + r, y, r);
        ctx.closePath();
    }

    // --- Public API ---

    update(dur, videoPath, fps, newEntries) {
        this.duration = dur;
        this.entries = newEntries || [];
        this.videoPath = videoPath || "";
        this.fps = (fps && fps > 0) ? fps : DEFAULT_FPS;
        this.playhead = this._snapSec(Math.max(0, Math.min(this.duration, this.playhead)));

        if (videoPath) {
            const url = api.apiURL("/foleytune/timeline_thumbnails?video_path=" +
                encodeURIComponent(videoPath));
            this.thumbImg.onerror = () =>
                console.warn("[FoleyTune timeline] thumbnail request failed (see Network tab):", url);
            if (this.thumbImg.src !== url) {
                this.thumbImg.src = url;
            }
            this._loadPreviewFrame();  // seed the scrub preview at the current playhead
        } else {
            console.warn("[FoleyTune timeline] no video_path in features — thumbnail strip will stay blank");
        }

        this._enforceWidth();
        this._updateCanvasSize();
        this.render();
        // Re-measure next frame in case the DOM widget hasn't settled its width.
        requestAnimationFrame(() => {
            this._enforceWidth();
            this._updateCanvasSize();
            this.render();
        });
    }

    restoreSegments(json) {
        try {
            const parsed = JSON.parse(json);
            if (Array.isArray(parsed)) this.segments = parsed;
        } catch (_) {}
    }

    // Recover segments from the most reliable source available, then re-sync
    // every store. Prefer node.properties (name-keyed, reorder-proof); fall back
    // to the widget value (for graphs saved before properties mirroring). If the
    // widget slot was scrambled, re-committing repairs it from the good source.
    _restore() {
        const props = this.node.properties && this.node.properties.foleytune_segments;
        for (const candidate of [props, this.segWidget && this.segWidget.value]) {
            if (!candidate || candidate === "[]") continue;
            try {
                const parsed = JSON.parse(candidate);
                if (Array.isArray(parsed) && parsed.length) {
                    this.segments = parsed;
                    this._commitFlush();  // heal widget + properties from the good copy
                    return;
                }
            } catch (_) { /* try next source */ }
        }
    }

    destroy() {
        this._closeMenu();
        this._resizeObserver?.disconnect();
        if (this._commitTimer) clearTimeout(this._commitTimer);
    }
}


app.registerExtension({
    name: "FoleyTune.LoRATimeline",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== "FoleyTuneLoRATimeline") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);
            const node = this;

            const container = document.createElement("div");
            // Always reserve the timeline height (shows an empty box before the
            // node runs — reverted from the collapse-when-empty behaviour).
            container.style.cssText =
                `width:100%;height:${TOTAL_H}px;display:flex;flex-direction:column;` +
                `overflow:hidden;user-select:none;`;

            const wHeight = () => TOTAL_H + 8;

            // Defer construction so widgets list is settled
            let editor = null;
            setTimeout(() => {
                editor = new TimelineEditor(node, container);
                // Robust restore: node.properties (reorder-proof) > widget value,
                // re-syncing both so a scrambled widget slot self-heals.
                editor._restore();
                node._timelineEditor = editor;
            }, 0);

            // Auto-populate: fill the timeline with 8s zones (entries in order;
            // SaFa-overlapping or contiguous per the auto_populate_safa toggle).
            this.addWidget("button", "Auto-populate 8s zones", null, () => {
                node._timelineEditor?._autoPopulate();
            });

            // DOM Widget. Use a neutral widget type (NOT "preview") — the
            // "preview" type makes ComfyUI apply image aspect-ratio sizing,
            // which computed the width from the height and collapsed the
            // timeline to ~half (esp. when the node was selected/re-laid-out).
            // A plain type is sized full-width like the textareas. We only
            // supply the height via getHeight.
            this.addDOMWidget("timeline", "foleytune_timeline", container, {
                serialize: false,
                hideOnZoom: false,
                getMinHeight: wHeight,
                getHeight: wHeight,
                getValue() { return ""; },
                setValue() {},
            });

            // Receive execution output
            const origOnExecuted = node.onExecuted;
            node.onExecuted = function (output) {
                origOnExecuted?.apply(this, arguments);
                if (!editor) return;

                const dur = output?.duration?.[0] ?? 0;
                const vpath = output?.video_path?.[0] ?? "";
                const fps = output?.fps?.[0] ?? 0;
                const ents = output?.entries?.[0] ?? [];

                editor.update(dur, vpath, fps, ents);
                node.setSize([node.size[0], node.computeSize([node.size[0], 0])[1]]);
                node?.graph?.setDirtyCanvas(true);
            };

            // Keep the timeline spanning the full node width when the node is
            // resized (the canvas re-measures clientWidth and redraws).
            const origOnResize = node.onResize;
            node.onResize = function (size) {
                origOnResize?.apply(this, arguments);
                editor?._onResize();
            };

            // Restore from saved workflow
            const origOnConfigure = node.onConfigure;
            node.onConfigure = function (info) {
                origOnConfigure?.apply(this, arguments);
                // segments_json will be restored by ComfyUI from widgets_values
                // editor.restoreSegments is called in the deferred init above
            };

            // Cleanup
            const origOnRemoved = node.onRemoved;
            node.onRemoved = function () {
                editor?.destroy();
                origOnRemoved?.apply(this, arguments);
            };
        };
    },
});


// Timeline Entry: fold the "inherit the sampler's global seed" choice INTO the
// native control_after_generate dropdown (instead of a duplicate mode widget).
// Picking "sampler" parks the seed at -1, the sentinel the backend reads as
// inherit; fixed/randomize/increment keep working on the number as usual.
app.registerExtension({
    name: "FoleyTune.LoRATimelineEntry",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== "FoleyTuneLoRATimelineEntry") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);
            const node = this;

            const wire = () => {
                const seedW = node.widgets?.find(w => w.name === "seed");
                const ctrlW = node.widgets?.find(w => w.name === "control_after_generate");
                if (!seedW || !ctrlW) return false;

                const vals = ctrlW.options?.values;
                if (Array.isArray(vals) && !vals.includes("sampler")) vals.push("sampler");

                if (!ctrlW._ftSamplerHook) {
                    ctrlW._ftSamplerHook = true;
                    const orig = ctrlW.callback;
                    ctrlW.callback = function (...args) {
                        const r = orig?.apply(this, args);
                        if (ctrlW.value === "sampler") {
                            seedW.value = -1;
                            seedW.callback?.call(seedW, -1);
                            node.graph?.setDirtyCanvas(true, true);
                        }
                        return r;
                    };
                    if (ctrlW.value === "sampler") seedW.value = -1;
                }
                return true;
            };

            // control_after_generate is appended a tick after the widgets are
            // built, so retry a few frames until it exists.
            let tries = 0;
            const tick = () => { if (!wire() && tries++ < 20) setTimeout(tick, 50); };
            setTimeout(tick, 0);
        };
    },
});
