<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Weed & Crop Detection</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <script>
        tailwind.config = {
            theme: {
                extend: {
                    colors: {
                        primary: '#10B981',
                        'primary-dark': '#059669',
                        secondary: '#3B82F6',
                        'secondary-dark': '#2563EB',
                        dark: '#1F2937',
                        light: '#F9FAFB',
                        crop: '#10B981',
                        weed: '#EF4444',
                    }
                }
            }
        }
    </script>
    <style>
        .detection-card {
            background-color: rgba(249, 250, 251, 0.8);
            backdrop-filter: blur(10px);
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1);
        }
        .glow-border {
            box-shadow: 0 0 15px rgba(16, 185, 129, 0.3);
        }
        .pulse {
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0% { transform: scale(1); box-shadow: 0 0 0 0 rgba(59, 130, 246, 0.4); }
            70% { transform: scale(1.02); box-shadow: 0 0 0 10px rgba(59, 130, 246, 0); }
            100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(59, 130, 246, 0); }
        }
    </style>
</head>
<body class="bg-gradient-to-br from-gray-50 to-gray-100 min-h-screen">
    <div class="max-w-7xl mx-auto px-4 py-8">
        <!-- Header -->
        <header class="text-center mb-12">
            <div class="flex items-center justify-center gap-4 mb-2">
                <i class="fas fa-leaf text-5xl text-primary"></i>
                <h1 class="text-4xl md:text-5xl font-bold text-dark">
                    Weed <span class="text-primary">&</span> Crop Detection
                </h1>
            </div>
            <p class="text-gray-600 max-w-2xl mx-auto text-lg">
                Use computer vision to identify weeds and crops in images, videos, or live camera feed. 
                Powered by YOLOv8 and Ultralytics.
            </p>
        </header>

        <div class="grid grid-cols-1 lg:grid-cols-4 gap-8">
            <!-- Sidebar -->
            <div class="lg:col-span-1">
                <div class="detection-card rounded-xl p-6 sticky top-8">
                    <h2 class="text-2xl font-bold text-dark mb-6 flex items-center gap-2">
                        <i class="fas fa-sliders-h text-secondary"></i>
                        Detection Options
                    </h2>
                    
                    <!-- Detection Mode -->
                    <div class="mb-6">
                        <h3 class="font-medium text-gray-700 mb-2">Detection Mode</h3>
                        <div class="space-y-3">
                            <div class="flex items-center p-3 rounded-lg border border-gray-200 hover:border-primary hover:bg-primary/5 cursor-pointer transition">
                                <input type="radio" id="image" name="mode" class="h-4 w-4 text-primary" checked>
                                <label for="image" class="ml-3 text-gray-700 cursor-pointer flex items-center gap-2">
                                    <i class="fas fa-image text-secondary"></i> Image
                                </label>
                            </div>
                            <div class="flex items-center p-3 rounded-lg border border-gray-200 hover:border-primary hover:bg-primary/5 cursor-pointer transition">
                                <input type="radio" id="video" name="mode" class="h-4 w-4 text-primary">
                                <label for="video" class="ml-3 text-gray-700 cursor-pointer flex items-center gap-2">
                                    <i class="fas fa-video text-secondary"></i> Video
                                </label>
                            </div>
                            <div class="flex items-center p-3 rounded-lg border border-gray-200 hover:border-primary hover:bg-primary/5 cursor-pointer transition">
                                <input type="radio" id="camera" name="mode" class="h-4 w-4 text-primary">
                                <label for="camera" class="ml-3 text-gray-700 cursor-pointer flex items-center gap-2">
                                    <i class="fas fa-camera text-secondary"></i> Live Camera
                                </label>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Confidence Threshold -->
                    <div class="mb-6">
                        <div class="flex justify-between items-center mb-2">
                            <h3 class="font-medium text-gray-700">Confidence Threshold</h3>
                            <span id="confidenceValue" class="bg-primary text-white font-medium px-2 py-1 rounded text-sm">0.3</span>
                        </div>
                        <input type="range" min="0" max="100" value="30" class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-primary">
                    </div>
                    
                    <!-- Legend -->
                    <div class="pt-4 border-t border-gray-200">
                        <h3 class="font-medium text-gray-700 mb-2">Detection Legend</h3>
                        <div class="space-y-2">
                            <div class="flex items-center">
                                <div class="w-3 h-3 rounded-full bg-crop mr-2"></div>
                                <span class="text-sm">Crop</span>
                            </div>
                            <div class="flex items-center">
                                <div class="w-3 h-3 rounded-full bg-weed mr-2"></div>
                                <span class="text-sm">Weed</span>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Status Card -->
                <div class="detection-card rounded-xl p-6 mt-6">
                    <h3 class="text-xl font-bold text-dark mb-4 flex items-center gap-2">
                        <i class="fas fa-info-circle text-blue-500"></i> System Status
                    </h3>
                    <div class="space-y-3">
                        <div class="flex items-center">
                            <div class="h-2 w-2 rounded-full bg-green-500 mr-3 animate-pulse"></div>
                            <span class="text-gray-700">Model: Loaded</span>
                        </div>
                        <div class="flex items-center">
                            <div class="h-2 w-2 rounded-full bg-green-500 mr-3 animate-pulse"></div>
                            <span class="text-gray-700">Detection: Ready</span>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Main Content -->
            <div class="lg:col-span-3">
                <!-- Image Detection -->
                <div id="imageSection" class="detection-card rounded-xl p-6 mb-8">
                    <div class="flex justify-between items-center mb-6">
                        <h2 class="text-2xl font-bold text-dark flex items-center gap-2">
                            <i class="fas fa-image text-secondary"></i> Image Detection
                        </h2>
                        <div class="relative">
                            <button class="bg-primary text-white font-medium px-4 py-2 rounded-lg hover:bg-primary-dark transition flex items-center gap-2">
                                <i class="fas fa-cloud-upload-alt"></i> Upload Image
                            </button>
                            <input type="file" class="absolute inset-0 opacity-0 cursor-pointer" accept="image/*">
                        </div>
                    </div>
                    
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <!-- Original Image -->
                        <div>
                            <h3 class="font-medium text-gray-700 mb-2 flex items-center gap-2">
                                <i class="fas fa-file-image text-gray-500"></i> Original Image
                            </h3>
                            <div class="border-2 border-dashed border-gray-300 rounded-xl h-64 md:h-72 flex items-center justify-center">
                                <div class="text-center">
                                    <i class="fas fa-image text-gray-300 text-4xl mb-2"></i>
                                    <p class="text-gray-400">Upload an image to get started</p>
                                </div>
                            </div>
                        </div>
                        
                        <!-- Processed Image -->
                        <div>
                            <h3 class="font-medium text-gray-700 mb-2 flex items-center gap-2">
                                <i class="fas fa-tags text-gray-500"></i> Processed Image
                            </h3>
                            <div class="border-2 border-dashed border-gray-300 rounded-xl h-64 md:h-72 flex items-center justify-center">
                                <div class="text-center">
                                    <i class="fas fa-project-diagram text-gray-300 text-4xl mb-2"></i>
                                    <p class="text-gray-400">Detection results will appear here</p>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="mt-6 flex items-center gap-4">
                        <button class="bg-secondary text-white font-medium px-5 py-2.5 rounded-lg hover:bg-secondary-dark transition flex items-center gap-2 pulse">
                            <i class="fas fa-search"></i> Detect Weeds & Crops
                        </button>
                        <div class="bg-yellow-50 border border-yellow-200 rounded-lg px-4 py-2 text-yellow-800 text-sm flex-1">
                            <i class="fas fa-info-circle mr-1"></i> Upload an image and click detect to process.
                        </div>
                    </div>
                    
                    <!-- Detection Stats -->
                    <div class="grid grid-cols-1 sm:grid-cols-2 gap-4 mt-6">
                        <div class="bg-green-50 border border-green-200 rounded-lg p-4">
                            <div class="flex items-center gap-3">
                                <div class="w-10 h-10 rounded-full bg-crop flex items-center justify-center">
                                    <i class="fas fa-leaf text-white"></i>
                                </div>
                                <div>
                                    <p class="text-gray-600 text-sm">Crop Detection</p>
                                    <p class="font-bold text-gray-800 text-xl">0</p>
                                </div>
                            </div>
                        </div>
                        <div class="bg-red-50 border border-red-200 rounded-lg p-4">
                            <div class="flex items-center gap-3">
                                <div class="w-10 h-10 rounded-full bg-weed flex items-center justify-center">
                                    <i class="fas fa-tree text-white"></i>
                                </div>
                                <div>
                                    <p class="text-gray-600 text-sm">Weed Detection</p>
                                    <p class="font-bold text-gray-800 text-xl">0</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Video Detection -->
                <div id="videoSection" class="detection-card rounded-xl p-6 mb-8 hidden">
                    <div class="flex justify-between items-center mb-6">
                        <h2 class="text-2xl font-bold text-dark flex items-center gap-2">
                            <i class="fas fa-video text-secondary"></i> Video Detection
                        </h2>
                        <div class="relative">
                            <button class="bg-primary text-white font-medium px-4 py-2 rounded-lg hover:bg-primary-dark transition flex items-center gap-2">
                                <i class="fas fa-cloud-upload-alt"></i> Upload Video
                            </button>
                            <input type="file" class="absolute inset-0 opacity-0 cursor-pointer" accept="video/*">
                        </div>
                    </div>
                    
                    <div class="border-2 border-dashed border-gray-300 rounded-xl h-96 flex items-center justify-center">
                        <div class="text-center">
                            <i class="fas fa-film text-gray-300 text-5xl mb-3"></i>
                            <p class="text-gray-600">Upload a video to process weed and crop detection</p>
                            <p class="text-sm text-gray-500 mt-2">Supports MP4, MOV, AVI formats</p>
                        </div>
                    </div>
                    
                    <div class="mt-6 flex items-center gap-4">
                        <button class="bg-secondary text-white font-medium px-5 py-2.5 rounded-lg hover:bg-secondary-dark transition flex items-center gap-2">
                            <i class="fas fa-play-circle"></i> Process Video
                        </button>
                        <div class="bg-blue-50 border border-blue-200 rounded-lg px-4 py-2 text-blue-800 text-sm flex-1">
                            <i class="fas fa-info-circle mr-1"></i> Processing may take time depending on video length.
                        </div>
                    </div>
                    
                    <div class="mt-6">
                        <div class="grid grid-cols-2 gap-4">
                            <div>
                                <h3 class="font-medium text-gray-700 mb-2">Processing Progress</h3>
                                <div class="h-2 bg-gray-200 rounded-full">
                                    <div class="h-2 bg-green-500 rounded-full w-0"></div>
                                </div>
                                <p class="text-gray-500 text-sm mt-2">0% completed</p>
                            </div>
                            <div>
                                <h3 class="font-medium text-gray-700 mb-2">Detection Stats</h3>
                                <div class="flex gap-4">
                                    <div class="text-center">
                                        <p class="text-gray-600 text-sm">Crops</p>
                                        <p class="font-bold text-gray-800 text-xl">0</p>
                                    </div>
                                    <div class="text-center">
                                        <p class="text-gray-600 text-sm">Weeds</p>
                                        <p class="font-bold text-gray-800 text-xl">0</p>
                                    </div>
                                    <div class="text-center">
                                        <p class="text-gray-600 text-sm">FPS</p>
                                        <p class="font-bold text-gray-800 text-xl">0</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Live Camera -->
                <div id="cameraSection" class="detection-card rounded-xl p-6 hidden">
                    <h2 class="text-2xl font-bold text-dark mb-6 flex items-center gap-2">
                        <i class="fas fa-camera text-secondary"></i> Live Camera Detection
                    </h2>
                    
                    <div class="rounded-2xl overflow-hidden mb-6 glow-border">
                        <div class="bg-black aspect-video flex items-center justify-center relative">
                            <div class="absolute w-full h-full opacity-20 bg-gradient-to-r from-green-400 to-blue-500"></div>
                            <div class="z-10 text-center">
                                <i class="fas fa-video-slash text-gray-300 text-6xl"></i>
                                <p class="text-gray-300 mt-2">Camera Feed</p>
                            </div>
                        </div>
                    </div>
                    
                    <div class="flex flex-wrap gap-4">
                        <button class="bg-green-600 text-white font-medium px-5 py-2.5 rounded-lg hover:bg-green-700 transition flex items-center gap-2">
                            <i class="fas fa-play"></i> Start Camera
                        </button>
                        <button class="bg-red-500 text-white font-medium px-5 py-2.5 rounded-lg hover:bg-red-600 transition flex items-center gap-2">
                            <i class="fas fa-stop"></i> Stop Detection
                        </button>
                        <button class="bg-gray-100 border border-gray-300 text-gray-700 font-medium px-5 py-2.5 rounded-lg hover:bg-gray-200 transition flex items-center gap-2">
                            <i class="fas fa-camera"></i> Capture Image
                        </button>
                    </div>
                    
                    <div class="mt-6">
                        <div class="grid grid-cols-2 gap-4">
                            <div>
                                <h3 class="font-medium text-gray-700 mb-2">Camera Status</h3>
                                <div class="flex items-center">
                                    <div class="h-2 w-2 rounded-full bg-red-500 mr-2"></div>
                                    <span class="text-gray-700">Not Active</span>
                                </div>
                            </div>
                            <div>
                                <h3 class="font-medium text-gray-700 mb-2">Active Detections</h3>
                                <div class="flex gap-4">
                                    <div class="text-center">
                                        <p class="text-gray-600 text-sm">Crops</p>
                                        <p class="font-bold text-gray-800 text-xl">0</p>
                                    </div>
                                    <div class="text-center">
                                        <p class="text-gray-600 text-sm">Weeds</p>
                                        <p class="font-bold text-gray-800 text-xl">0</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="mt-8 bg-blue-50 border border-blue-200 rounded-lg p-4">
                        <div class="flex items-start gap-3">
                            <i class="fas fa-info-circle text-blue-500 text-xl mt-1"></i>
                            <p class="text-blue-800">
                                Click "Start Camera" and grant camera access to begin real-time weed and crop detection. 
                                The system will process each frame and overlay detection information in real-time.
                            </p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Footer -->
        <footer class="mt-12 pt-6 border-t border-gray-200 text-center text-gray-600">
            <div class="flex items-center justify-center gap-4 mb-3">
                <div class="bg-gray-100 border border-gray-200 rounded-lg py-2 px-4 font-medium">
                    <span class="text-primary font-bold">YOLO</span>v8
                </div>
                <div class="bg-gray-100 border border-gray-200 rounded-lg py-2 px-4 font-medium">
                    <span class="text-blue-500 font-bold">Streamlit</span>
                </div>
            </div>
            <p>
                Developed by <span class="text-primary font-medium">blurerjr/mu</span> using Ultralytics YOLO and Streamlit.
            </p>
        </footer>
    </div>

    <script>
        // Toggle between detection modes
        document.querySelectorAll('input[name="mode"]').forEach(radio => {
            radio.addEventListener('change', function() {
                document.getElementById('imageSection').classList.add('hidden');
                document.getElementById('videoSection').classList.add('hidden');
                document.getElementById('cameraSection').classList.add('hidden');
                
                if (this.id === 'image') {
                    document.getElementById('imageSection').classList.remove('hidden');
                } else if (this.id === 'video') {
                    document.getElementById('videoSection').classList.remove('hidden');
                } else if (this.id === 'camera') {
                    document.getElementById('cameraSection').classList.remove('hidden');
                }
            });
        });
        
        // Update confidence value display
        const confidenceSlider = document.querySelector('input[type="range"]');
        const confidenceValue = document.getElementById('confidenceValue');
        
        confidenceSlider.addEventListener('input', function() {
            const value = this.value / 100;
            confidenceValue.textContent = value.toFixed(2);
        });
        
        // Add animation to detection button
        const detectBtn = document.querySelector('.pulse');
        detectBtn.addEventListener('mouseover', function() {
            this.classList.add('animate-none');
        });
        
        detectBtn.addEventListener('mouseout', function() {
            this.classList.remove('animate-none');
        });
        
        // Handle file selection
        document.querySelectorAll('input[type="file"]').forEach(input => {
            input.addEventListener('change', function() {
                if (this.files.length > 0) {
                    const fileName = this.files[0].name;
                    const button = this.previousElementSibling;
                    button.innerHTML = `<i class="fas fa-check mr-2"></i> ${fileName}`;
                    button.classList.add('bg-green-100', 'text-green-800');
                    
                    setTimeout(() => {
                        button.classList.remove('bg-green-100', 'text-green-800');
                        button.innerHTML = '<i class="fas fa-cloud-upload-alt"></i> Upload Image';
                    }, 3000);
                }
            });
        });
    </script>
</body>
</html>
