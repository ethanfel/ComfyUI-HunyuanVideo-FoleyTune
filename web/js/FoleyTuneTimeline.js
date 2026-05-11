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
const HANDLE_W = 6;
const CANVAS_H = RULER_H + TRACK_H;
const TOTAL_H = THUMB_H + CANVAS_H;
const MIN_SEG_SEC = 0.5;
const SNAP = 0.5;

function snap(sec) {
    return Math.round(sec / SNAP) * SNAP;
}

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

        // Find and hide the segments_json widget
        this.segWidget = node.widgets?.find(w => w.name === "segments_json");
        if (this.segWidget) {
            this.segWidget.type = "converted-widget";
            this.segWidget.computeSize = () => [0, -4];
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
        this.thumbImg.style.cssText = "height:100%;position:absolute;left:0;top:0;";
        this.thumbStrip.appendChild(this.thumbImg);
        this.container.appendChild(this.thumbStrip);

        // Canvas for ruler + segments
        this.canvas = document.createElement("canvas");
        this.canvas.style.cssText = `width:100%;height:${CANVAS_H}px;display:block;`;
        this.container.appendChild(this.canvas);
        this.ctx = this.canvas.getContext("2d");
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
        return this.duration > 0 ? snap(Math.max(0, Math.min(this.duration, (x / this._canvasW()) * this.duration))) : 0;
    }

    _pointerPos(e) {
        const r = this.canvas.getBoundingClientRect();
        return { x: e.clientX - r.left, y: e.clientY - r.top };
    }

    _hitTest(x, y) {
        if (y < RULER_H || y > RULER_H + TRACK_H) return { idx: -1, edge: null };
        // Check segments in reverse (last rendered = on top)
        for (let i = this.segments.length - 1; i >= 0; i--) {
            const seg = this.segments[i];
            const left = this._secToX(seg.start_sec);
            const right = this._secToX(seg.end_sec);
            if (x >= left && x <= right) {
                const edge = (x - left < HANDLE_W) ? "left" :
                             (right - x < HANDLE_W) ? "right" : null;
                return { idx: i, edge };
            }
        }
        return { idx: -1, edge: null };
    }

    // --- Events ---

    _onPointerDown(e) {
        e.stopPropagation();
        const { x, y } = this._pointerPos(e);
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
            const dSec = this._xToSec(x) - this._xToSec(this.drag.startX);
            const seg = this.segments[this.drag.idx];

            if (this.drag.type === "move") {
                const len = this.drag.origEnd - this.drag.origStart;
                const ns = snap(Math.max(0, Math.min(this.duration - len, this.drag.origStart + dSec)));
                seg.start_sec = ns;
                seg.end_sec = ns + len;
            } else if (this.drag.type === "resize-left") {
                seg.start_sec = snap(Math.max(0, Math.min(seg.end_sec - MIN_SEG_SEC, this.drag.origStart + dSec)));
            } else if (this.drag.type === "resize-right") {
                seg.end_sec = snap(Math.max(seg.start_sec + MIN_SEG_SEC, Math.min(this.duration, this.drag.origEnd + dSec)));
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

        if (hit.edge) {
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
            this.drag = null;
            this.canvas.releasePointerCapture(e.pointerId);
            this._commitFlush();
            this.render();
        }
    }

    _onPointerLeave() {
        if (!this.drag) {
            this.hoverIdx = -1;
            this.hoverEdge = null;
            this.canvas.style.cursor = "default";
            this.render();
        }
    }

    _onDblClick(e) {
        e.stopPropagation();
        e.preventDefault();
        if (!this.entries.length || this.duration <= 0) return;

        const { x } = this._pointerPos(e);
        const sec = this._xToSec(x);

        let entryIdx = 0;
        if (this.entries.length > 1) {
            const labels = this.entries.map((en, i) => `${i}: ${en.label}`).join("\n");
            const input = prompt(`Select LoRA entry:\n${labels}`, "0");
            if (input === null) return;
            entryIdx = parseInt(input, 10);
            if (entryIdx < 0 || entryIdx >= this.entries.length) return;
        }

        const startSec = snap(Math.max(0, sec - 2));
        const endSec = snap(Math.min(this.duration, sec + 2));
        this.segments.push({
            entry_index: entryIdx,
            start_sec: startSec,
            end_sec: endSec,
            strength: this.entries[entryIdx].strength,
        });
        this.segments.sort((a, b) => a.start_sec - b.start_sec);
        this.selectedIdx = this.segments.length - 1;
        this._commitFlush();
        this.render();
    }

    _onContextMenu(e) {
        e.preventDefault();
        e.stopPropagation();
        const { x, y } = this._pointerPos(e);
        const hit = this._hitTest(x, y);
        if (hit.idx >= 0) {
            this.segments.splice(hit.idx, 1);
            this.selectedIdx = -1;
            this._commitFlush();
            this.render();
        }
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
        if (this.segWidget) {
            this.segWidget.value = JSON.stringify(this.segments);
        }
    }

    // --- Resize ---

    _onResize() {
        this._updateCanvasSize();
        this.render();
    }

    _updateCanvasSize() {
        const dpr = this._dpr();
        const w = this._canvasW();
        this.canvas.width = w * dpr;
        this.canvas.height = CANVAS_H * dpr;
        this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        // Scale thumbnail to match
        this.thumbImg.style.width = this.thumbStrip.clientWidth + "px";
    }

    // --- Rendering ---

    render() {
        const ctx = this.ctx;
        const w = this._canvasW();
        if (w <= 0 || this.duration <= 0) return;

        ctx.clearRect(0, 0, w, CANVAS_H);

        this._drawRuler(ctx, w);
        this._drawTrack(ctx, w);
        this._drawSegments(ctx, w);
    }

    _drawRuler(ctx, w) {
        ctx.fillStyle = "#1a1a2e";
        ctx.fillRect(0, 0, w, RULER_H);

        ctx.fillStyle = "#666";
        ctx.font = "10px monospace";

        const dur = this.duration;
        // Pick nice tick interval
        const pxPerSec = w / dur;
        const minTickPx = 40;
        const rawStep = minTickPx / pxPerSec;
        const niceSteps = [0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60];
        const step = niceSteps.find(s => s >= rawStep) || 60;

        for (let t = 0; t <= dur; t += step) {
            const x = this._secToX(t);
            const isMajor = step < 1 ? (Math.round(t * 10) % (Math.round(step * 50))) === 0
                                     : (t % (step * 5)) === 0;

            ctx.fillStyle = "#555";
            ctx.fillRect(x, 0, 1, isMajor ? RULER_H - 4 : RULER_H - 10);

            if (isMajor || pxPerSec * step > 50) {
                ctx.fillStyle = "#999";
                const label = t < 60 ? `${t.toFixed(step < 1 ? 1 : 0)}s`
                                     : `${Math.floor(t/60)}m${(t%60).toFixed(0).padStart(2,"0")}`;
                ctx.fillText(label, x + 3, RULER_H - 3);
            }
        }

        // Bottom border
        ctx.fillStyle = "#333";
        ctx.fillRect(0, RULER_H - 1, w, 1);
    }

    _drawTrack(ctx, w) {
        ctx.fillStyle = "#0e0e1a";
        ctx.fillRect(0, RULER_H, w, TRACK_H);

        // Grid lines at each ruler tick for alignment reference
        ctx.fillStyle = "#1a1a2e";
        const dur = this.duration;
        const pxPerSec = w / dur;
        const minTickPx = 40;
        const rawStep = minTickPx / pxPerSec;
        const niceSteps = [0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60];
        const step = niceSteps.find(s => s >= rawStep) || 60;
        for (let t = 0; t <= dur; t += step) {
            ctx.fillRect(this._secToX(t), RULER_H, 1, TRACK_H);
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

    _drawSegment(ctx, i) {
        const seg = this.segments[i];
        const entry = this.entries[seg.entry_index] || {};
        const color = resolveColor(entry);
        const label = entry.label || `LoRA ${seg.entry_index}`;

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

            // Time range
            ctx.font = "9px monospace";
            ctx.fillStyle = "#ccc";
            ctx.shadowBlur = 0;
            ctx.fillText(`${seg.start_sec.toFixed(1)}s–${seg.end_sec.toFixed(1)}s`, centerX, centerY + 8);

            ctx.restore();
        }

        // Resize handle indicators
        if (isHover || isSelected) {
            ctx.fillStyle = "rgba(255,255,255,0.4)";
            ctx.fillRect(x + 1, y + h / 4, 2, h / 2);
            ctx.fillRect(x2 - 3, y + h / 4, 2, h / 2);
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

    update(dur, videoPath, newEntries) {
        this.duration = dur;
        this.entries = newEntries || [];

        if (videoPath) {
            const url = api.apiURL("/foleytune/timeline_thumbnails?video_path=" +
                encodeURIComponent(videoPath));
            if (this.thumbImg.src !== url) {
                this.thumbImg.src = url;
            }
        }

        this._updateCanvasSize();
        this.render();
    }

    restoreSegments(json) {
        try {
            const parsed = JSON.parse(json);
            if (Array.isArray(parsed)) this.segments = parsed;
        } catch (_) {}
    }

    destroy() {
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
            container.style.cssText =
                "width:100%;display:flex;flex-direction:column;user-select:none;";

            // Defer construction so widgets list is settled
            let editor = null;
            setTimeout(() => {
                editor = new TimelineEditor(node, container);

                // Restore from saved workflow
                const segWidget = node.widgets?.find(w => w.name === "segments_json");
                if (segWidget?.value && segWidget.value !== "[]") {
                    editor.restoreSegments(segWidget.value);
                }

                node._timelineEditor = editor;
            }, 0);

            // DOM Widget
            const widget = this.addDOMWidget("timeline", "preview", container, {
                serialize: false,
                hideOnZoom: false,
                getMinHeight: () => TOTAL_H + 8,
                getHeight: () => TOTAL_H + 8,
                getValue() { return ""; },
                setValue() {},
            });
            widget.computeSize = function (width) {
                return [width, editor?.duration > 0 ? TOTAL_H + 8 : -4];
            };

            // Receive execution output
            const origOnExecuted = node.onExecuted;
            node.onExecuted = function (output) {
                origOnExecuted?.apply(this, arguments);
                if (!editor) return;

                const dur = output?.duration?.[0] ?? 0;
                const vpath = output?.video_path?.[0] ?? "";
                const ents = output?.entries?.[0] ?? [];

                editor.update(dur, vpath, ents);
                node.setSize([node.size[0], node.computeSize([node.size[0], 0])[1]]);
                node?.graph?.setDirtyCanvas(true);
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
