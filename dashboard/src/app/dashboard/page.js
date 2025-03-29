"use client";

import { useEffect, useRef, useState } from "react";
import { 
	VideoCameraIcon, 
	ArrowLeftIcon,
	ChartBarIcon,
	ClockIcon,
	CpuChipIcon,
	EyeIcon
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

		// Clean up on unmount
		return () => {
			if (socketRef.current) {
				socketRef.current.disconnect();
			}
		};
	}, []);

	return (
		<div className="flex min-h-screen bg-gradient-to-b from-navy-900 to-navy-800 p-4">
			{/* Left sidebar for stats */}
			<div className="w-1/4 p-4 overflow-y-auto space-y-4">
				<StatisticsCard 
					title="Detection Stats" 
					icon={<ChartBarIcon className="h-6 w-6 text-blue-400" />}
					stats={[
						{ 
							label: "Total Detections", 
							value: detectionStats.total_detections.toLocaleString()
						},
						{ 
							label: "FPS", 
							value: detectionStats.fps 
						},
						{ 
							label: "Connection", 
							value: socketConnected ? "Connected" : "Disconnected",
							color: socketConnected ? "text-green-400" : "text-red-400"
						}
					]}
				/>
				
				<ClassDistributionCard 
					title="Object Classes" 
					icon={<EyeIcon className="h-6 w-6 text-purple-400" />}
					classCounts={detectionStats.class_counts} 
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
			<div className="w-1/4 p-4 overflow-y-auto">
				<RecentDetectionsCard 
					title="Recent Detections" 
					icon={<ClockIcon className="h-6 w-6 text-teal-400" />}
					recentDetections={detectionStats.recent_detections} 
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

// Card component for displaying class distribution
function ClassDistributionCard({ title, icon, classCounts }) {
	// Convert object to sorted array for rendering
	const sortedClasses = Object.entries(classCounts || {})
		.sort((a, b) => b[1] - a[1])
		.slice(0, 8); // Show only top 8 classes
	
	// Calculate total for percentage
	const total = Object.values(classCounts || {}).reduce((sum, count) => sum + count, 0);
	
	return (
		<div className="bg-navy-700 rounded-lg shadow-md p-6 border border-navy-600">
			<div className="flex items-center mb-4">
				{icon}
				<h2 className="text-xl font-bold ml-2">{title}</h2>
			</div>
			
			{sortedClasses.length > 0 ? (
				<div className="space-y-3">
					{sortedClasses.map(([className, count], index) => {
						// Generate a deterministic color based on class name
						const hashValue = className.split('').reduce(
							(hash, char) => char.charCodeAt(0) + ((hash << 5) - hash), 0
						);
						const hue = Math.abs(hashValue % 360);
						const color = `hsl(${hue}, 70%, 60%)`;
						
						// Calculate percentage
						const percentage = total > 0 ? Math.round((count / total) * 100) : 0;
						
						return (
							<div key={index} className="space-y-1">
								<div className="flex justify-between items-center">
									<span className="text-navy-200 capitalize">{className}</span>
									<span className="text-sm font-medium">
										{count} <span className="text-navy-400">({percentage}%)</span>
									</span>
								</div>
								<div className="w-full bg-navy-800 rounded-full h-2.5">
									<div 
										className="h-2.5 rounded-full" 
										style={{ 
											width: `${percentage}%`,
											backgroundColor: color
										}}
									></div>
								</div>
							</div>
						);
					})}
				</div>
			) : (
				<p className="text-navy-300 text-center py-4">No objects detected yet</p>
			)}
		</div>
	);
}

// Card component for displaying recent detections
function RecentDetectionsCard({ title, icon, recentDetections }) {
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
		<div className="bg-navy-700 rounded-lg shadow-md p-6 border border-navy-600 h-full">
			<div className="flex items-center mb-4">
				{icon}
				<h2 className="text-xl font-bold ml-2">{title}</h2>
			</div>
			
			{recentDetections && recentDetections.length > 0 ? (
				<div className="space-y-4 max-h-[calc(100vh-200px)] overflow-y-auto pr-2">
					{recentDetections.map((detection, index) => (
						<div 
							key={index} 
							className="bg-navy-800 rounded-lg p-3 border border-navy-700 transition-all hover:border-navy-500"
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
							<div className="mt-1 bg-navy-900 rounded-lg p-2 text-xs">
								<div className="relative w-full h-[40px] border border-navy-700 rounded">
									{/* Position indicator */}
									<div 
										className="absolute w-2 h-2 rounded-full z-10"
										style={{ 
											left: `${detection.position.x * 100}%`, 
											top: `${detection.position.y * 100}%`,
											backgroundColor: detection.color,
											transform: 'translate(-50%, -50%)'
										}}
									></div>
									<div className="absolute inset-0 flex items-center justify-center text-[10px] text-navy-400">
										Position map
									</div>
								</div>
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