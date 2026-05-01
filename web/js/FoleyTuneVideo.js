import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const VIDEO_EXTENSIONS = ["webm", "mp4", "mkv", "gif", "mov", "avi"];

function addVideoPreview(nodeType) {
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
        onNodeCreated?.apply(this, arguments);

        const node = this;
        const videoEl = document.createElement("video");
        videoEl.controls = true;
        videoEl.loop = true;
        videoEl.muted = true;
        videoEl.style.width = "100%";
        videoEl.style.display = "none";
        videoEl.style.verticalAlign = "top";
        videoEl.style.objectFit = "contain";
        videoEl.style.background = "transparent";
        videoEl.onmouseenter = () => { videoEl.muted = false; };
        videoEl.onmouseleave = () => { videoEl.muted = true; };

        const previewWidget = this.addDOMWidget("video_preview", "preview", videoEl, {
            serialize: false,
            hideOnZoom: false,
        });

        node._ftVideoPreview = { videoEl, previewWidget };

        // Update preview after execution
        const onExecuted = node.onExecuted;
        node.onExecuted = function (output) {
            onExecuted?.apply(this, arguments);
            if (output?.gifs?.[0]) {
                const g = output.gifs[0];
                const params = new URLSearchParams({
                    filename: g.filename,
                    type: g.type || "temp",
                    subfolder: g.subfolder || "",
                });
                videoEl.src = api.apiURL("/view?" + params.toString());
                videoEl.style.display = "block";
            }
        };
    };
}

function addUploadWidget(nodeType) {
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
        onNodeCreated?.apply(this, arguments);

        const node = this;
        const pathWidget = this.widgets.find((w) => w.name === "video");
        if (!pathWidget) return;

        const fileInput = document.createElement("input");
        fileInput.type = "file";
        fileInput.accept = "video/*,image/gif";
        fileInput.style.display = "none";
        document.body.appendChild(fileInput);

        fileInput.onchange = async () => {
            if (!fileInput.files.length) return;
            const file = fileInput.files[0];
            const body = new FormData();
            body.append("image", file);
            body.append("overwrite", "true");
            const resp = await api.fetchApi("/upload/image", { method: "POST", body });
            if (resp.ok) {
                const data = await resp.json();
                if (!pathWidget.options.values.includes(data.name)) {
                    pathWidget.options.values.push(data.name);
                }
                pathWidget.value = data.name;
                pathWidget.callback?.(data.name);
            }
        };

        const uploadWidget = this.addWidget("button", "choose video to upload", null, () => {
            fileInput.click();
        });
        uploadWidget.serialize = false;

        // Drag-drop support
        this.onDragOver = (e) => !!e?.dataTransfer?.types?.includes?.("Files");
        this.onDragDrop = async (e) => {
            const file = e?.dataTransfer?.files?.[0];
            if (!file) return false;
            const ext = file.name.split(".").pop()?.toLowerCase();
            if (!VIDEO_EXTENSIONS.includes(ext)) return false;
            const body = new FormData();
            body.append("image", file);
            body.append("overwrite", "true");
            const resp = await api.fetchApi("/upload/image", { method: "POST", body });
            if (resp.ok) {
                const data = await resp.json();
                if (!pathWidget.options.values.includes(data.name)) {
                    pathWidget.options.values.push(data.name);
                }
                pathWidget.value = data.name;
                pathWidget.callback?.(data.name);
            }
            return true;
        };

        // Preview on combo selection change
        const origCallback = pathWidget.callback;
        pathWidget.callback = function (value) {
            origCallback?.apply(this, arguments);
            if (!value) return;
            const preview = node._ftVideoPreview;
            if (preview) {
                const params = new URLSearchParams({
                    filename: value,
                    type: "input",
                    subfolder: "",
                });
                preview.videoEl.src = api.apiURL("/view?" + params.toString());
                preview.videoEl.style.display = "block";
            }
        };
    };
}

app.registerExtension({
    name: "FoleyTune.VideoNodes",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name === "FoleyTuneVideoLoader") {
            addVideoPreview(nodeType);
        }
        if (nodeData?.name === "FoleyTuneVideoLoaderUpload") {
            addVideoPreview(nodeType);
            addUploadWidget(nodeType);
        }
        if (nodeData?.name === "FoleyTuneVideoCombiner") {
            addVideoPreview(nodeType);
        }
    },
});
