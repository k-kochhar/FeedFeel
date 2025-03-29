"use client";

import { useEffect, useRef, useState } from "react";
import { VideoCameraIcon, ArrowLeftIcon } from "@heroicons/react/24/outline";
import Link from "next/link";

export default function Dashboard() {
	return (
		<div className="flex min-h-screen bg-gradient-to-b from-navy-900 to-navy-800 p-4">
			{/* Left sidebar for cards */}
			<div className="w-1/4 p-4 overflow-y-auto">
				<div className="rounded-lg shadow-md p-6 mb-4 border">
					<h2 className="text-xl font-bold mb-4 text-black">Card Title</h2>
					<p className="text-black">
						This is a placeholder card on the left side. You can configure this
						later.
					</p>
				</div>
				<div className="bg-navy-700 rounded-lg shadow-md p-6 border border-navy-600">
					<h2 className="text-xl font-bold mb-4">Another Card</h2>
					<p className="text-navy-200">
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
				<div className="bg-navy-700 rounded-lg shadow-md p-6 mb-4 border border-navy-600">
					<h2 className="text-xl font-bold mb-4">Right Card</h2>
					<p className="text-navy-200">
						This is a placeholder card on the right side. You can configure this
						later.
					</p>
				</div>
				<div className="bg-navy-700 rounded-lg shadow-md p-6 border border-navy-600">
					<h2 className="text-xl font-bold mb-4">More Info</h2>
					<p className="text-navy-200">
						Additional placeholder content for the right sidebar.
					</p>
				</div>
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