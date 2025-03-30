"use client";

import { useEffect, useRef, useState } from "react";
import { 
	VideoCameraIcon, 
	ArrowLeftIcon,
	ChartBarIcon,
	ClockIcon,
	CpuChipIcon,
	EyeIcon,
	ArrowPathIcon,
	BeakerIcon
} from "@heroicons/react/24/outline";
import Link from "next/link";
import { io } from "socket.io-client";

export default function Dashboard() {
	const [detectionStats, setDetectionStats] = useState({
		total_detections: 0,
		class_counts: {},
		recent_detections: [],
		fps: 0
	});
	const [embeddingData, setEmbeddingData] = useState({
		object: "",
		raw_embedding: [],
		pooled_embedding: [],
		audio_signal: [],
		stepper_pattern: [],
		processing_time: 0
	});
	const [socketConnected, setSocketConnected] = useState(false);
	const socketRef = useRef(null);

	// Connect to WebSocket
	useEffect(() => {
		// Initialize socket connection
		socketRef.current = io("http://localhost:5001");

		// Handle socket connection events
		socketRef.current.on("connect", () => {
			console.log("Connected to WebSocket server");
			setSocketConnected(true);
		});

		socketRef.current.on("disconnect", () => {
			console.log("Disconnected from WebSocket server");
			setSocketConnected(false);
		});

		// Listen for detection stats updates
		socketRef.current.on("detection_stats", (data) => {
			setDetectionStats(data);
		});
		
		// Listen for embedding visualization data
		socketRef.current.on("embedding_visualization", (data) => {
			console.log("Received embedding visualization data", data);
			setEmbeddingData(data);
		});

		// Clean up on unmount
		return () => {
			if (socketRef.current) {
				socketRef.current.disconnect();
			}
		};
	}, []);
	
	// Function to request new embedding visualization
	const requestNewVisualization = (objectName = null) => {
		if (socketRef.current && socketRef.current.connected) {
			socketRef.current.emit("request_embedding_viz", { object: objectName });
		}
	};

	return (
		<div className="flex min-h-screen bg-gradient-to-b from-navy-900 to-navy-800 p-4">
			{/* Left sidebar for stats */}
			<div className="w-1/4 p-4 overflow-y-auto space-y-4">
				<EmbeddingVisualizerCard 
					title="Embedding Visualizer" 
					icon={<BeakerIcon className="h-6 w-6 text-purple-400" />}
					embeddingData={embeddingData}
					onRefresh={requestNewVisualization}
				/>
			</div>

			{/* Center area for camera feed */}
			<div className="w-2/4 p-4 flex flex-col items-center justify-start">
				<h1 className="text-2xl font-bold mb-4">Camera Feed</h1>
				<div className="relative w-full h-full flex justify-center items-start">
					<CameraFeed />
				</div>
			</div>

			{/* Right sidebar for recent detections */}
			<div className="w-1/4 p-4 overflow-y-auto space-y-4">
				<StatisticsCard 
					title="Detection Status" 
					icon={<ChartBarIcon className="h-6 w-6 text-blue-400" />}
					stats={[
						{ 
							label: "Connection", 
							value: socketConnected ? "Connected" : "Disconnected",
							color: socketConnected ? "text-green-400" : "text-red-400"
						}
					]}
				/>
				<RecentDetectionsCard 
					title="Recent Detections" 
					icon={<ClockIcon className="h-6 w-6 text-teal-400" />}
					recentDetections={detectionStats.recent_detections} 
					onSelectObject={requestNewVisualization}
				/>
			</div>
		</div>
	);
}

// CameraFeed component for handling video
function CameraFeed() {
	const containerRef = useRef(null);
	const [error, setError] = useState(null);
	const [isStreamConnected, setIsStreamConnected] = useState(false);
	const [streamUrl, setStreamUrl] = useState("http://localhost:5001/video_feed");

	// Effect to check if the stream server is running
	useEffect(() => {
		const checkStreamConnection = async () => {
			try {
				// Try to connect to the stream server root to verify it's running
				const response = await fetch("http://localhost:5001/");
				if (response.ok) {
					setIsStreamConnected(true);
					setError(null);
				} else {
					setIsStreamConnected(false);
					setError("Stream server is running but returned an error");
				}
			} catch (err) {
				console.error("Error connecting to stream server:", err);
				setIsStreamConnected(false);
				setError("Cannot connect to YOLO stream server. Make sure it's running on port 5001");
			}
		};

		// Check connection immediately
		checkStreamConnection();

		// Set up periodic checking
		const interval = setInterval(checkStreamConnection, 5000);

		// Cleanup on unmount
		return () => clearInterval(interval);
	}, []);

	return (
		<div className="flex flex-col items-center">
			{error && (
				<div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4 w-full">
					{error}
					<p className="text-sm mt-1">
						Run &apos;python logic/mjpeg_stream.py&apos; to start the stream server.
					</p>
				</div>
			)}

			<div
				ref={containerRef}
				className="rounded-2xl overflow-hidden shadow-lg flex justify-center items-center bg-navy-950"
				style={{
					width: "650px",
					height: "800px",
					maxWidth: "100%",
					maxHeight: "calc(100vh - 150px)",
				}}
			>
				{isStreamConnected ? (
					<img
						src={streamUrl}
						alt="YOLO Detection Stream"
						className="w-auto h-auto object-contain"
						style={{
							maxHeight: "100%",
							maxWidth: "100%",
						}}
					/>
				) : (
					<div className="text-center p-8 text-navy-300">
						<VideoCameraIcon className="mx-auto h-12 w-12 text-navy-400 mb-4" />
						<h3 className="text-lg font-medium">Stream not connected</h3>
						<p className="mt-2">
							Start the YOLO detection stream server to view the feed.
						</p>
						<p className="mt-4 text-xs text-navy-400">
							Run &apos;python logic/mjpeg_stream.py&apos; in your terminal
						</p>
					</div>
				)}
			</div>

			<div className="mt-4 text-xs text-navy-300">
				{isStreamConnected
					? "Connected to YOLO detection stream"
					: "Waiting for stream server connection..."}
			</div>
		</div>
	);
}

// Card component for displaying statistics
function StatisticsCard({ title, icon, stats }) {
	return (
		<div className="bg-navy-700 rounded-lg shadow-md p-6 border border-navy-600">
			<div className="flex items-center mb-4">
				{icon}
				<h2 className="text-xl font-bold ml-2">{title}</h2>
			</div>
			<div className="space-y-4">
				{stats.map((stat, index) => (
					<div key={index} className="flex justify-between items-center">
						<span className="text-navy-300">{stat.label}</span>
						<span className={`font-semibold text-lg ${stat.color || 'text-white'}`}>
							{stat.value}
						</span>
					</div>
				))}
			</div>
		</div>
	);
}

// Card component for visualizing the embedding transformation process
function EmbeddingVisualizerCard({ title, icon, embeddingData, onRefresh }) {
	const chartHeight = 60; // Height for each visualization chart
	
	// Function to get a gradient color based on value
	const getGradientColor = (value, min, max) => {
		// Normalize value between 0 and 1
		const normalized = (value - min) / (max - min);
		
		// Create a gradient from blue to purple to red
		let r, g, b;
		if (normalized < 0.5) {
			// Blue to purple
			r = Math.round(normalized * 2 * 255);
			g = 0;
			b = 255;
		} else {
			// Purple to red
			r = 255;
			g = 0;
			b = Math.round((1 - (normalized - 0.5) * 2) * 255);
		}
		
		return `rgb(${r}, ${g}, ${b})`;
	};
	
	// Function to generate bars for visualization
	const generateBars = (data, label) => {
		if (!data || data.length === 0) return null;
		
		// Get min and max values for scaling
		const min = Math.min(...data);
		const max = Math.max(...data);
		
		return (
			<div className="mb-4">
				<div className="flex justify-between mb-1">
					<span className="text-sm text-navy-300">{label}</span>
					<span className="text-xs text-navy-400">{data.length} values</span>
				</div>
				<div className="relative h-[60px] bg-navy-800 rounded overflow-hidden">
					{data.map((value, index) => {
						// Calculate normalized height
						const height = ((value - min) / (max - min || 1)) * 100;
						// Calculate width based on number of items
						const width = 100 / Math.min(data.length, 100);
						
						// Only show a subset of bars if there are too many
						if (data.length > 100 && index > 100) return null;
						
						return (
							<div
								key={index}
								className="absolute bottom-0"
								style={{
									left: `${index * width}%`,
									width: `${width}%`,
									height: `${height}%`,
									backgroundColor: getGradientColor(value, min, max),
									opacity: 0.8
								}}
							/>
						);
					})}
				</div>
			</div>
		);
	};
	
	return (
		<div className="bg-navy-700 rounded-lg shadow-md p-6 border border-navy-600">
			<div className="flex justify-between items-center mb-4">
				<div className="flex items-center">
					{icon}
					<h2 className="text-xl font-bold ml-2">{title}</h2>
				</div>
				<button 
					onClick={() => onRefresh()}
					className="p-1.5 bg-navy-600 hover:bg-navy-500 rounded-full transition-colors"
					title="Generate new visualization"
				>
					<ArrowPathIcon className="h-4 w-4 text-navy-300" />
				</button>
			</div>
			
			{embeddingData?.object ? (
				<div className="space-y-4">
					<div className="flex justify-between">
						<span className="text-lg font-medium">
							&quot;{embeddingData.object}&quot;
						</span>
						<span className="text-sm text-navy-300">
							{embeddingData.processing_time}s
						</span>
					</div>
					
					<div className="bg-navy-800 rounded p-3 space-y-4 max-h-[500px] overflow-y-auto scrollbar-thin">
						<div className="text-center space-y-1">
							<h3 className="text-sm font-medium">Transformation Process</h3>
							<p className="text-xs text-navy-400">
								Text → Embedding → Pooling → FFT → Stepper Pattern
							</p>
						</div>
						
						{/* Visualize raw embedding */}
						{generateBars(embeddingData.raw_embedding, "OpenAI Text Embedding")}
						
						{/* Arrow indicator */}
						<div className="flex justify-center">
							<ArrowPathIcon className="h-4 w-4 text-navy-400 animate-spin-slow" />
						</div>
						
						{/* Visualize pooled embedding */}
						{generateBars(embeddingData.pooled_embedding, "Pooled Embedding")}
						
						{/* Arrow indicator */}
						<div className="flex justify-center">
							<ArrowPathIcon className="h-4 w-4 text-navy-400 animate-spin-slow" />
						</div>
						
						{/* Visualize audio signal (FFT output) */}
						{generateBars(embeddingData.audio_signal, "Inverse FFT Signal")}
						
						{/* Arrow indicator */}
						<div className="flex justify-center">
							<ArrowPathIcon className="h-4 w-4 text-navy-400 animate-spin-slow" />
						</div>
						
						{/* Visualize stepper pattern */}
						{generateBars(embeddingData.stepper_pattern, "Stepper Motor Pattern")}
					</div>
					
					<div className="text-center text-xs text-navy-400 mt-2">
						The math behind haptic feedback generation
					</div>
				</div>
			) : (
				<div className="flex flex-col items-center justify-center h-64 text-navy-400">
					<BeakerIcon className="h-12 w-12 mb-4 opacity-30" />
					<p>No embedding data yet</p>
					<p className="text-xs mt-2">
						Waiting for detection or click refresh
					</p>
				</div>
			)}
		</div>
	);
}

// Card component for displaying recent detections
function RecentDetectionsCard({ title, icon, recentDetections, onSelectObject }) {
	// Function to format timestamp
	const formatTimeAgo = (timestamp) => {
		const now = Date.now() / 1000;
		const secondsAgo = Math.floor(now - timestamp);
		
		if (secondsAgo < 5) return "just now";
		if (secondsAgo < 60) return `${secondsAgo}s ago`;
		if (secondsAgo < 3600) return `${Math.floor(secondsAgo / 60)}m ago`;
		return `${Math.floor(secondsAgo / 3600)}h ago`;
	};
	
	return (
		<div className="bg-navy-700 rounded-lg shadow-md p-6 border border-navy-600 h-fit">
			<div className="flex items-center mb-4">
				{icon}
				<h2 className="text-xl font-bold ml-2">{title}</h2>
			</div>
			
			{recentDetections && recentDetections.length > 0 ? (
				<div className="space-y-4 max-h-[calc(80vh-200px)] overflow-y-auto pr-2 scrollbar-thin">
					{recentDetections.map((detection, index) => (
						<div 
							key={index} 
							className="bg-navy-800 rounded-lg p-3 border border-navy-700 transition-all hover:border-navy-500 cursor-pointer"
							onClick={() => onSelectObject(detection.class_name)}
						>
							<div className="flex items-center justify-between">
								<div className="flex items-center">
									<div 
										className="w-3 h-3 rounded-full mr-2" 
										style={{ backgroundColor: detection.color }}
									></div>
									<span className="font-medium capitalize">{detection.class_name}</span>
								</div>
								<span className="text-xs text-navy-400">
									{formatTimeAgo(detection.timestamp)}
								</span>
							</div>
							<div className="mt-2 text-sm flex justify-between">
								<span className="text-navy-300">Confidence: <span className="text-navy-200">{Math.round(detection.confidence * 100)}%</span></span>
								<span className="text-navy-300">Size: <span className="text-navy-200">{(detection.size * 100).toFixed(1)}%</span></span>
							</div>
					
						</div>
					))}
				</div>
			) : (
				<div className="flex flex-col items-center justify-center h-64 text-navy-400">
					<EyeIcon className="h-12 w-12 mb-4 opacity-30" />
					<p>No recent detections</p>
				</div>
			)}
		</div>
	);
}