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

            <!-- Video element used for BOTH webcam playback and file playback -->
            <video ref="videoElement" 
                   class="w-full h-full object-contain" 
                   :class="{ 'hidden': !activeMode }"
                   playsinline 
                   muted 
                   controls 
                   v-show="activeMode"></video>
            
            <!-- Hidden canvas for capturing webcam frames -->
            <canvas ref="hiddenCanvas" class="hidden"></canvas>
            
            <!-- SVG overlay for drawing markers -->
            <svg class="absolute top-0 left-0 w-full h-full pointer-events-none" v-if="activeMode">
                <!-- Dummy skeleton markers -->
                <circle v-for="(marker, i) in markers" :key="i" :cx="marker.x" :cy="marker.y" r="6" fill="#3b82f6" stroke="white" stroke-width="2" />
                <!-- Dummy skeleton lines -->
                <line v-if="markers.length >= 2" :x1="markers[0].x" :y1="markers[0].y" :x2="markers[1].x" :y2="markers[1].y" stroke="#3b82f6" stroke-width="3" />
            </svg>

            <!-- Reset overlay button -->
            <button v-if="activeMode" @click="stop" class="absolute top-4 right-4 bg-red-600/80 hover:bg-red-700 text-white px-3 py-1 rounded text-sm shadow backdrop-blur-sm transition">
                Close Media
            </button>
        </div>
    `,
    data() {
        return {
            activeMode: null, // 'webcam' or 'file'
            stream: null,
            markers: [],
            animationFrameId: null
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
                this.activeMode = 'webcam';
                
                // Start capturing frames
                this.captureFrame();
                // Add dummy markers for webcam mode just to show it works
                this.syncMarkers();
            } catch (err) {
                console.error("Error accessing webcam:", err);
                this.$emit('error', 'Could not access webcam');
            }
        },
        handleFileUpload(event) {
            const file = event.target.files[0];
            if (file) {
                const url = URL.createObjectURL(file);
                const video = this.$refs.videoElement;
                video.srcObject = null;
                video.src = url;
                video.setAttribute('controls', 'true');
                video.load();
                this.activeMode = 'file';
                
                // Trigger backend processing
                this.$emit('file_uploaded', { name: file.name, size: file.size });
                
                // Sync dummy markers with playback
                this.syncMarkers();
                
                // Reset file input so same file can be uploaded again if needed
                event.target.value = '';
            }
        },
        captureFrame() {
            if (this.activeMode !== 'webcam') return;
            
            const video = this.$refs.videoElement;
            const canvas = this.$refs.hiddenCanvas;
            
            if (video.readyState === video.HAVE_ENOUGH_DATA) {
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                
                // Emit frame to backend (base64 string)
                const dataUrl = canvas.toDataURL('image/jpeg', 0.8);
                this.$emit('frame_ready', dataUrl);
            }
            
            // Loop at ~10 FPS for backend processing
            setTimeout(() => this.captureFrame(), 100);
        },
        syncMarkers() {
            if (!this.activeMode) return;
            
            const video = this.$refs.videoElement;
            if (video && video.readyState >= video.HAVE_METADATA) {
                // Ensure video dimensions are known to map coordinates correctly
                const rect = video.getBoundingClientRect();
                
                // Dummy logic: animate 2 connected markers bouncing around
                const time = performance.now() / 1000;
                
                // Center of the video element
                const cx = rect.width / 2;
                const cy = rect.height / 2;
                const radius = Math.min(cx, cy) * 0.5;
                
                this.markers = [
                    { x: cx + Math.sin(time) * radius, y: cy + Math.cos(time) * radius },
                    { x: cx + Math.cos(time * 1.5) * radius, y: cy + Math.sin(time * 1.5) * radius }
                ];
            }
            
            this.animationFrameId = requestAnimationFrame(this.syncMarkers);
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
        }
    },
    beforeUnmount() {
        this.stop();
    }
}
