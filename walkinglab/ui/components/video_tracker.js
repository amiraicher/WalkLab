export default {
    template: `
        <div class="video-tracker w-full h-full relative bg-slate-900 rounded-lg overflow-hidden flex flex-col items-center justify-center">
            
            <div v-if="!activeMode" class="flex flex-col items-center gap-6 p-8">
                <div class="text-white text-xl mb-4 font-semibold">Select Media Source</div>
                <div class="flex gap-4">
                    <button @click="startWebcam" class="px-6 py-3 bg-blue-600 text-white rounded-lg shadow hover:bg-blue-700 transition flex items-center gap-2">
                        <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                            <path d="M2 6a2 2 0 012-2h12a2 2 0 012 2v8a2 2 0 01-2 2H4a2 2 0 01-2-2V6z" />
                        </svg>
                        Start Camera
                    </button>
                    <div class="relative">
                        <button class="px-6 py-3 bg-teal-600 text-white rounded-lg shadow hover:bg-teal-700 transition flex items-center gap-2" @click="$refs.fileInput.click()">
                            <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                                <path fill-rule="evenodd" d="M3 17a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zM6.293 6.707a1 1 0 010-1.414l3-3a1 1 0 011.414 0l3 3a1 1 0 01-1.414 1.414L11 5.414V13a1 1 0 11-2 0V5.414L7.707 6.707a1 1 0 01-1.414 0z" clip-rule="evenodd" />
                            </svg>
                            Upload Video
                        </button>
                        <input type="file" ref="fileInput" class="hidden" accept="video/*" @change="handleFileUpload">
                    </div>
                </div>
            </div>

            <!-- Video element used for live webcam capture (hidden in webcam mode) and visible file playback -->
            <video ref="videoElement" 
                   class="w-full h-full object-contain" 
                   :class="{ 'hidden': activeMode !== 'file' }"
                   playsinline 
                   muted 
                   controls 
                   v-show="activeMode === 'file'"></video>

            <!-- Buffered display canvas for 200ms delayed real-time webcam playback -->
            <canvas ref="displayCanvas"
                    class="w-full h-full object-contain"
                    :class="{ 'hidden': activeMode !== 'webcam' }"
                    v-show="activeMode === 'webcam'"></canvas>
            
            <!-- Hidden canvas for capturing and encoding webcam frames -->
            <canvas ref="hiddenCanvas" class="hidden"></canvas>
            
            <!-- SVG overlay for drawing markers -->
            <svg class="absolute top-0 left-0 w-full h-full pointer-events-none" v-if="activeMode">
                <!-- Skeleton markers -->
                <circle v-for="(marker, i) in markers" :key="marker.id !== undefined ? marker.id : i" :cx="marker.x" :cy="marker.y" r="6" fill="#3b82f6" stroke="white" stroke-width="2" />
                <!-- Skeleton lines -->
                <line v-if="markers.length >= 2" :x1="markers[0].x" :y1="markers[0].y" :x2="markers[1].x" :y2="markers[1].y" stroke="#3b82f6" stroke-width="3" />
            </svg>

            <!-- Reset overlay button -->
            <button v-if="activeMode" @click="stop" class="absolute top-4 right-4 bg-red-600/80 hover:bg-red-700 text-white px-3 py-1 rounded text-sm shadow backdrop-blur-sm transition z-10">
                Close Media
            </button>

            <!-- Progress Overlay -->
            <div v-if="progressStage" class="absolute inset-0 flex items-center justify-center bg-black/70 z-50 backdrop-blur-sm">
                <div class="bg-slate-800 p-6 rounded-xl shadow-2xl w-80 text-center border border-slate-700">
                    <div class="text-white text-lg font-semibold mb-4">{{ progressStage }}</div>
                    <div class="w-full bg-slate-700 rounded-full h-3 mb-2 overflow-hidden">
                        <div class="bg-blue-500 h-3 rounded-full transition-all duration-300 ease-out" :style="{ width: progressPercent + '%' }"></div>
                    </div>
                    <div class="text-sm text-slate-400">{{ Math.round(progressPercent) }}%</div>
                </div>
            </div>
        </div>
    `,
    data() {
        return {
            activeMode: null, // 'webcam' or 'file'
            stream: null,
            markers: [],
            all_frames_markers: [],
            animationFrameId: null,
            isProcessing: false,
            progressStage: '',
            progressPercent: 0,
            estimatedSeconds: 0,
            // Real-time 200ms buffer, frame indexing, dilation, and interpolation state
            bufferDelayMs: 200,
            frameDilation: 4,
            interpolationMode: 'linear',
            frameIndex: 0,
            lastInferredFrameIndex: -4,
            frameBuffer: [],
            frameMarkersMap: {},
            inferredKeyframes: [],
            displayedFrameIndex: -1
        }
    },
    methods: {
        async startWebcam() {
            try {
                this.stream = await navigator.mediaDevices.getUserMedia({ video: true });
                const video = this.$refs.videoElement;
                video.srcObject = this.stream;
                video.removeAttribute('controls'); // hide controls for live stream
                await video.play();

                this.frameIndex = 0;
                this.lastInferredFrameIndex = -this.frameDilation;
                this.frameBuffer = [];
                this.frameMarkersMap = {};
                this.inferredKeyframes = [];
                this.displayedFrameIndex = -1;
                this.markers = [];
                this.activeMode = 'webcam';
                
                // Start capturing frames into the 200ms buffer and dispatching dilated inference
                this.captureFrame();
                // Start synchronized delayed playback and marker interpolation loop
                this.syncMarkers();
            } catch (err) {
                console.error("Error accessing webcam:", err);
                this.$emit('error', 'Could not access webcam');
            }
        },
        async handleFileUpload(event) {
            const file = event.target.files[0];
            if (file) {
                const url = URL.createObjectURL(file);
                const video = this.$refs.videoElement;
                video.srcObject = null;
                video.src = url;
                video.setAttribute('controls', 'true');
                video.load();
                this.activeMode = 'file';
                this.all_frames_markers = [];
                
                this.$emit('file_uploaded_start', { name: file.name });
                
                video.onloadedmetadata = async () => {
                    let duration = video.duration;
                    if (!duration || isNaN(duration) || !isFinite(duration)) {
                        duration = 10; // robust fallback
                    }
                    const assumedFps = 30;
                    const totalFrames = Math.floor(duration * assumedFps);
                    
                    try {
                        const estRes = await fetch(`/api/estimate?frames=${totalFrames}`);
                        const estData = await estRes.json();
                        this.estimatedSeconds = estData.estimated_seconds;
                    } catch (e) {
                        this.estimatedSeconds = totalFrames / 5.23;
                    }
                    
                    this.progressStage = 'Uploading video...';
                    this.progressPercent = 0;
                    
                    const xhr = new XMLHttpRequest();
                    xhr.open('POST', '/api/upload_video', true);
                    
                    let analyzingStarted = false;
                    
                    xhr.upload.onprogress = (e) => {
                        if (e.lengthComputable && this.progressStage === 'Uploading video...') {
                            this.progressPercent = (e.loaded / e.total) * 100;
                            
                            // Transition to Analyzing perfectly when upload hits 100%
                            if (e.loaded >= e.total && !analyzingStarted) {
                                analyzingStarted = true;
                                this.progressStage = 'Analyzing pose...';
                                this.progressPercent = 0;
                                
                                const startTime = performance.now();
                                const durationMs = (this.estimatedSeconds || 10) * 1000;
                                
                                const animateProgress = (now) => {
                                    if (this.progressStage !== 'Analyzing pose...') return;
                                    
                                    const elapsed = now - startTime;
                                    let percent = (elapsed / durationMs) * 100;
                                    if (percent > 99) percent = 99; // Hold at 99%
                                    
                                    this.progressPercent = percent;
                                    requestAnimationFrame(animateProgress);
                                };
                                requestAnimationFrame(animateProgress);
                            }
                        }
                    };
                    
                    xhr.onload = () => {
                        this.progressStage = '';
                        if (xhr.status === 200) {
                            const markers = JSON.parse(xhr.responseText);
                            this.all_frames_markers = markers;
                            this.$emit('file_uploaded_success', { name: file.name });
                        } else {
                            console.error('Upload failed');
                            this.$emit('error', 'Video processing failed');
                        }
                    };
                    
                    xhr.onerror = () => {
                        this.progressStage = '';
                        console.error('Network error during upload');
                        this.$emit('error', 'Network error during upload');
                    };
                    
                    const formData = new FormData();
                    formData.append('file', file);
                    xhr.send(formData);
                };
                
                this.syncMarkers();
                event.target.value = '';
            }
        },
        captureFrame() {
            if (this.activeMode !== 'webcam') return;
            
            const video = this.$refs.videoElement;
            const canvas = this.$refs.hiddenCanvas;
            
            if (video && video.readyState === video.HAVE_ENOUGH_DATA && video.videoWidth > 0) {
                const currentIdx = this.frameIndex++;
                const now = performance.now();

                // Store frame snapshot in the real-time display buffer
                const frameCanvas = document.createElement('canvas');
                frameCanvas.width = video.videoWidth;
                frameCanvas.height = video.videoHeight;
                const frameCtx = frameCanvas.getContext('2d');
                frameCtx.drawImage(video, 0, 0, frameCanvas.width, frameCanvas.height);

                this.frameBuffer.push({
                    frameIndex: currentIdx,
                    timestamp: now,
                    canvas: frameCanvas
                });

                // Cap buffer size to avoid unbounded memory growth
                if (this.frameBuffer.length > 180) {
                    this.frameBuffer.shift();
                }

                // Frame dilation: only dispatch every N-th frame when backend is idle
                const shouldInfer = (currentIdx - this.lastInferredFrameIndex) >= this.frameDilation;
                if (shouldInfer && !this.isProcessing) {
                    canvas.width = video.videoWidth;
                    canvas.height = video.videoHeight;
                    const ctx = canvas.getContext('2d');
                    ctx.drawImage(frameCanvas, 0, 0);

                    const dataUrl = canvas.toDataURL('image/jpeg', 0.8);
                    this.isProcessing = true;
                    this.lastInferredFrameIndex = currentIdx;
                    this.$emit('frame_ready', {
                        frame_index: currentIdx,
                        frame_data: dataUrl,
                        buffer_delay_ms: this.bufferDelayMs,
                        dilation_factor: this.frameDilation,
                        interpolation_method: this.interpolationMode
                    });
                }
            }
            
            setTimeout(() => this.captureFrame(), 33);
        },
        calculateVideoBounds() {
            const video = this.$refs.videoElement;
            const displayEl = this.activeMode === 'webcam' ? this.$refs.displayCanvas : video;
            if (!displayEl || !video || !video.videoWidth || !video.videoHeight) {
                return { actualWidth: 0, actualHeight: 0, offsetX: 0, offsetY: 0 };
            }
            const rect = displayEl.getBoundingClientRect();
            const videoRatio = video.videoWidth / video.videoHeight;
            const containerRatio = rect.width / rect.height;
            
            let actualWidth, actualHeight, offsetX, offsetY;
            
            if (videoRatio > containerRatio) {
                actualWidth = rect.width;
                actualHeight = rect.width / videoRatio;
                offsetX = 0;
                offsetY = (rect.height - actualHeight) / 2;
            } else {
                actualHeight = rect.height;
                actualWidth = rect.height * videoRatio;
                offsetX = (rect.width - actualWidth) / 2;
                offsetY = 0;
            }
            
            return { actualWidth, actualHeight, offsetX, offsetY };
        },
        interpolateMarkersLinear(startMarkers, endMarkers, alpha) {
            if (!startMarkers || startMarkers.length === 0) {
                return alpha >= 0.5 ? (endMarkers || []) : [];
            }
            if (!endMarkers || endMarkers.length === 0) {
                return alpha < 0.5 ? startMarkers : [];
            }

            const clampedAlpha = Math.max(0.0, Math.min(1.0, alpha));
            const hasIds = startMarkers[0].id !== undefined || endMarkers[0].id !== undefined;

            if (hasIds) {
                const startById = {};
                const endById = {};
                const allIds = [];

                startMarkers.forEach(m => {
                    if (m.id !== undefined) {
                        startById[m.id] = m;
                        allIds.push(m.id);
                    }
                });
                endMarkers.forEach(m => {
                    if (m.id !== undefined) {
                        endById[m.id] = m;
                        if (startById[m.id] === undefined) {
                            allIds.push(m.id);
                        }
                    }
                });

                const result = [];
                for (const id of allIds) {
                    const m0 = startById[id];
                    const m1 = endById[id];
                    if (m0 && m1) {
                        result.push({
                            id: id,
                            x: m0.x + clampedAlpha * (m1.x - m0.x),
                            y: m0.y + clampedAlpha * (m1.y - m0.y)
                        });
                    } else if (m0 && clampedAlpha < 0.5) {
                        result.push({ id: id, x: m0.x, y: m0.y });
                    } else if (m1 && clampedAlpha >= 0.5) {
                        result.push({ id: id, x: m1.x, y: m1.y });
                    }
                }
                return result;
            }

            const count = Math.min(startMarkers.length, endMarkers.length);
            const result = [];
            for (let i = 0; i < count; i++) {
                result.push({
                    x: startMarkers[i].x + clampedAlpha * (endMarkers[i].x - startMarkers[i].x),
                    y: startMarkers[i].y + clampedAlpha * (endMarkers[i].y - startMarkers[i].y)
                });
            }
            return result;
        },
        getMarkersForFrame(targetFrameIdx) {
            if (this.frameMarkersMap[targetFrameIdx] !== undefined) {
                return this.frameMarkersMap[targetFrameIdx];
            }

            if (this.inferredKeyframes.length === 0) {
                return [];
            }

            if (this.interpolationMode === 'none' || this.inferredKeyframes.length === 1) {
                let latest = this.inferredKeyframes[0];
                for (const kf of this.inferredKeyframes) {
                    if (kf.frameIndex <= targetFrameIdx) {
                        latest = kf;
                    } else {
                        break;
                    }
                }
                return latest.markers || [];
            }

            let prevKf = null;
            let nextKf = null;
            for (const kf of this.inferredKeyframes) {
                if (kf.frameIndex <= targetFrameIdx) {
                    prevKf = kf;
                } else if (kf.frameIndex > targetFrameIdx) {
                    nextKf = kf;
                    break;
                }
            }

            if (prevKf && nextKf && nextKf.frameIndex > prevKf.frameIndex) {
                const alpha = (targetFrameIdx - prevKf.frameIndex) / (nextKf.frameIndex - prevKf.frameIndex);
                return this.interpolateMarkersLinear(prevKf.markers, nextKf.markers, alpha);
            }

            if (prevKf) {
                return prevKf.markers || [];
            }
            return nextKf ? (nextKf.markers || []) : [];
        },
        syncMarkers() {
            if (!this.activeMode) return;
            
            if (this.activeMode === 'webcam') {
                const now = performance.now();
                let readyIdx = -1;

                for (let i = 0; i < this.frameBuffer.length; i++) {
                    if (now - this.frameBuffer[i].timestamp >= this.bufferDelayMs) {
                        readyIdx = i;
                    } else {
                        break;
                    }
                }

                if (readyIdx >= 0) {
                    const entry = this.frameBuffer[readyIdx];
                    this.frameBuffer.splice(0, readyIdx);

                    const displayCanvas = this.$refs.displayCanvas;
                    if (displayCanvas && entry.canvas) {
                        if (displayCanvas.width !== entry.canvas.width || displayCanvas.height !== entry.canvas.height) {
                            displayCanvas.width = entry.canvas.width;
                            displayCanvas.height = entry.canvas.height;
                        }
                        const ctx = displayCanvas.getContext('2d');
                        ctx.drawImage(entry.canvas, 0, 0);
                        this.displayedFrameIndex = entry.frameIndex;

                        const bounds = this.calculateVideoBounds();
                        const rawMarkers = this.getMarkersForFrame(entry.frameIndex);
                        this.markers = rawMarkers.map(m => ({
                            id: m.id,
                            x: bounds.offsetX + (m.x * bounds.actualWidth),
                            y: bounds.offsetY + (m.y * bounds.actualHeight)
                        }));
                    }
                }
            } else if (this.activeMode === 'file' && this.all_frames_markers && this.all_frames_markers.length > 0) {
                const video = this.$refs.videoElement;
                if (video && video.readyState >= video.HAVE_METADATA && video.videoWidth > 0) {
                    const bounds = this.calculateVideoBounds();
                    
                    const frameIndex = Math.floor(video.currentTime * 30);
                    if (frameIndex < this.all_frames_markers.length) {
                        const markers = this.all_frames_markers[frameIndex] || [];
                        this.markers = markers.map(m => ({
                            id: m.id,
                            x: bounds.offsetX + (m.x * bounds.actualWidth),
                            y: bounds.offsetY + (m.y * bounds.actualHeight)
                        }));
                    }
                }
            }
            
            this.animationFrameId = requestAnimationFrame(this.syncMarkers);
        },
        update_markers(payload) {
            this.isProcessing = false;
            if (!payload) return;

            // Legacy array mode fallback
            if (Array.isArray(payload)) {
                const video = this.$refs.videoElement;
                if (video && video.readyState >= video.HAVE_METADATA && video.videoWidth > 0) {
                    const bounds = this.calculateVideoBounds();
                    this.markers = payload.map(m => ({
                        id: m.id,
                        x: bounds.offsetX + (m.x * bounds.actualWidth),
                        y: bounds.offsetY + (m.y * bounds.actualHeight)
                    }));
                }
                return;
            }

            // Indexed real-time payload mode
            if (payload.buffer_delay_ms && payload.buffer_delay_ms !== this.bufferDelayMs) {
                if (payload.buffer_adjusted) {
                    console.warn(
                        `[VideoTracker] Inference time (${(payload.inference_time_ms || 0).toFixed(1)} ms) ` +
                        `exceeded half of buffer delay (${this.bufferDelayMs} ms). ` +
                        `Adjusting buffer delay to ${payload.buffer_delay_ms} ms.`
                    );
                }
                this.bufferDelayMs = payload.buffer_delay_ms;
            }

            if (payload.dilation_factor) {
                this.frameDilation = payload.dilation_factor;
            }
            if (payload.interpolation_method) {
                this.interpolationMode = payload.interpolation_method;
            }

            if (payload.interpolated_frames) {
                for (const [k, v] of Object.entries(payload.interpolated_frames)) {
                    this.frameMarkersMap[ parseInt(k, 10) ] = v;
                }
            }

            if (payload.frame_index !== undefined && Array.isArray(payload.markers)) {
                const fIdx = parseInt(payload.frame_index, 10);
                this.frameMarkersMap[fIdx] = payload.markers;
                this.inferredKeyframes.push({
                    frameIndex: fIdx,
                    markers: payload.markers
                });
                this.inferredKeyframes.sort((a, b) => a.frameIndex - b.frameIndex);
                if (this.inferredKeyframes.length > 60) {
                    this.inferredKeyframes.shift();
                }

                // Prune old frameMarkersMap entries
                const minKeepIdx = fIdx - 300;
                for (const key of Object.keys(this.frameMarkersMap)) {
                    if (parseInt(key, 10) < minKeepIdx) {
                        delete this.frameMarkersMap[key];
                    }
                }
            }
        },
        set_tracking_config(config) {
            if (!config) return;
            if (config.bufferDelayMs !== undefined && config.bufferDelayMs > 0) {
                this.bufferDelayMs = config.bufferDelayMs;
            }
            if (config.frameDilation !== undefined && config.frameDilation >= 1) {
                this.frameDilation = config.frameDilation;
            }
            if (config.interpolationMode !== undefined) {
                this.interpolationMode = config.interpolationMode;
            }
        },
        set_batch_markers(all_markers) {
            this.all_frames_markers = all_markers;
        },
        stop() {
            if (this.stream) {
                this.stream.getTracks().forEach(track => track.stop());
                this.stream = null;
            }
            if (this.animationFrameId) {
                cancelAnimationFrame(this.animationFrameId);
                this.animationFrameId = null;
            }
            if (this.$refs.videoElement) {
                this.$refs.videoElement.pause();
                this.$refs.videoElement.removeAttribute('src');
                this.$refs.videoElement.load();
            }
            this.activeMode = null;
            this.markers = [];
            this.frameBuffer = [];
            this.frameMarkersMap = {};
            this.inferredKeyframes = [];
            this.displayedFrameIndex = -1;
        }
    },
    beforeUnmount() {
        this.stop();
    }
}
