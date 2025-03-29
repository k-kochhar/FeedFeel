"use client";

import { useEffect, useRef, useState } from "react";

export default function Home() {
  return (
    <div className="flex min-h-screen bg-gray-100 dark:bg-gray-900 p-4">
      {/* Left sidebar for cards */}
      <div className="w-1/4 p-4 overflow-y-auto">
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6 mb-4">
          <h2 className="text-xl font-bold mb-4">Card Title</h2>
          <p className="text-gray-600 dark:text-gray-300">
            This is a placeholder card on the left side. You can configure this later.
          </p>
        </div>
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6">
          <h2 className="text-xl font-bold mb-4">Another Card</h2>
          <p className="text-gray-600 dark:text-gray-300">
            More placeholder content for demonstration.
          </p>
        </div>
      </div>

      {/* Center area for camera feed */}
      <div className="w-2/4 p-4 flex flex-col items-center justify-start">
        <h1 className="text-2xl font-bold mb-4">Camera Feed</h1>
        <div className="relative w-full h-full flex justify-center items-start">
          <CameraFeed />
        </div>
      </div>

      {/* Right sidebar for cards */}
      <div className="w-1/4 p-4 overflow-y-auto">
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6 mb-4">
          <h2 className="text-xl font-bold mb-4">Right Card</h2>
          <p className="text-gray-600 dark:text-gray-300">
            This is a placeholder card on the right side. You can configure this later.
          </p>
        </div>
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6">
          <h2 className="text-xl font-bold mb-4">More Info</h2>
          <p className="text-gray-600 dark:text-gray-300">
            Additional placeholder content for the right sidebar.
          </p>
        </div>
      </div>
    </div>
  );
}

// CameraFeed component for handling video
function CameraFeed() {
  const videoRef = useRef(null);
  const containerRef = useRef(null);
  const [availableCameras, setAvailableCameras] = useState([]);
  const [error, setError] = useState(null);
  const [isVirtualCamera, setIsVirtualCamera] = useState(true);
  const [videoDimensions, setVideoDimensions] = useState({ width: 0, height: 0 });

  // Event handler to get video dimensions
  const handleVideoMetadata = () => {
    if (videoRef.current) {
      const { videoWidth, videoHeight } = videoRef.current;
      setVideoDimensions({ width: videoWidth, height: videoHeight });
      console.log(`Video dimensions: ${videoWidth}x${videoHeight}`);
    }
  };

  useEffect(() => {
    // Get list of available cameras
    async function getAvailableCameras() {
      try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoDevices = devices.filter(device => device.kind === 'videoinput');
        setAvailableCameras(videoDevices);
      } catch (err) {
        console.error('Error getting devices:', err);
        setError('Failed to get camera devices');
      }
    }

    getAvailableCameras();
  }, []);

  useEffect(() => {
    async function setupCamera() {
      try {
        // Stop any existing stream
        if (videoRef.current && videoRef.current.srcObject) {
          const tracks = videoRef.current.srcObject.getTracks();
          tracks.forEach(track => track.stop());
        }

        setError(null);

        // Try to find the virtual camera by name
        const virtualCameraPatterns = [
          /virtual/i, /cam link/i, /obs/i, /droidcam/i, /iriun/i, /phone/i, /mobile/i
        ];
        
        const virtualCam = availableCameras.find(device => 
          virtualCameraPatterns.some(pattern => pattern.test(device.label))
        );
        
        // Find system camera as fallback
        const systemCam = availableCameras.find(device => 
          !(/virtual|obs|droidcam|iriun|cam link/i.test(device.label))
        );
        
        // Use virtual camera if available, otherwise use system camera
        const selectedCamera = virtualCam || systemCam;
        setIsVirtualCamera(!!virtualCam && selectedCamera === virtualCam);
        
        if (!selectedCamera) {
          setError('No camera detected. Please connect a camera.');
          return;
        }

        const constraints = {
          video: {
            deviceId: { exact: selectedCamera.deviceId },
            // Don't specify dimensions to get native camera resolution
          }
        };

        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (err) {
        console.error("Error accessing camera:", err);
        setError(`Failed to access camera: ${err.message}`);
      }
    }
    
    if (availableCameras.length > 0) {
      setupCamera();
    }
    
    // Cleanup function
    return () => {
      if (videoRef.current && videoRef.current.srcObject) {
        const tracks = videoRef.current.srcObject.getTracks();
        tracks.forEach(track => track.stop());
      }
    };
  }, [availableCameras]);

  return (
    <div className="flex flex-col items-center">
      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4 w-full">
          {error}
        </div>
      )}
      
      <div 
        ref={containerRef}
        className="rounded-2xl overflow-hidden shadow-lg flex justify-center items-center"
        style={{ 
          width: '650px',
          height: '1000px',
          maxWidth: '100%',
          maxHeight: 'calc(100vh - 150px)'
        }}
      >
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          onLoadedMetadata={handleVideoMetadata}
          className="w-auto h-auto"
          style={{ 
            transform: isVirtualCamera ? 'rotate(90deg)' : 'none',
            transformOrigin: 'center center',
            objectFit: 'contain',
            width: 'auto',
            height: isVirtualCamera ? '90%' : 'auto',
            maxHeight: '100%',
          }}
        />
      </div>
      
      {videoDimensions.width > 0 && (
        <div className="mt-2 text-xs text-gray-500">
          Camera resolution: {videoDimensions.width}x{videoDimensions.height}
        </div>
      )}
    </div>
  );
}
