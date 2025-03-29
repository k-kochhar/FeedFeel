"use client";

import { useState } from "react";
import Link from "next/link";
import { ArrowRightIcon, VideoCameraIcon, ShieldCheckIcon, BellAlertIcon, EyeIcon } from "@heroicons/react/24/outline";
import Image from "next/image";
import Glasses from "@/components/Glasses";

export default function Home() {
  return (
    <div className="bg-background min-h-screen">
      <div className="max-w-5xl mx-auto px-6 py-24">
        {/* Header */}
        <header className="mb-24">
          <div className="flex items-center gap-3">
            <div className="h-13 w-13 bg-accent/20 rounded-full flex items-center justify-center shadow-soft">
              <Glasses size="sm" />
            </div>
            <h1 className="text-2xl font-medium">SixthSense</h1>
          </div>
        </header>

        {/* Hero Section */}
        <section className="mb-32 flex flex-col md:flex-row gap-12">
          <div className="md:w-2/3">
            <h2 className="text-4xl md:text-5xl font-medium leading-tight mb-8">
              Transforming<br/> Vision into Touch
            </h2>
            
            <p className="text-base leading-relaxed mb-12 text-text/80">
              SixthSense uses smart glasses and real-time object detection 
              to convert visual data into vibrations, giving users a natural 
              and confident way to navigate their surroundings.
            </p>
            
            <div className="flex flex-col sm:flex-row gap-6">
              <Link
                href="/dashboard"
                className="group flex items-center justify-center rounded-xl bg-accent px-8 py-3.5 text-text font-medium"
              >
                <span>Go to Dashboard</span>
                <ArrowRightIcon className="ml-2 h-5 w-5 group-hover:translate-x-1 transition-transform" />
              </Link>
              <a
                href="#features"
                className="rounded-xl px-8 py-3.5 text-text/70 font-medium bg-background border border-text/10 hover:bg-gray-200 transition-all duration-300"
              >
                Learn More
              </a>
            </div>
          </div>
          
          <div className="md:w-1/3 flex items-center justify-center rounded-2xl">
              {/* Vibration visualization */}
              <div className="relative z-10 flex flex-col items-center justify-center py-6 -mt-12 -mr-24">
                <Image src="/glasses.svg" alt="glasses" width={400} height={400} />
              </div>
          </div>
        </section>

        {/* Features */}
        <section id="features" className="mb-32">
          <h2 className="text-2xl font-medium mb-16 text-center">Key Features</h2>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-12">
            <div className="bg-white rounded-2xl p-8 shadow-soft transition-all duration-300 hover:shadow-medium">
              <div className="h-12 w-12 bg-accent/20 rounded-xl flex items-center justify-center mb-6">
                <Glasses size="sm" />
              </div>
              <h3 className="text-xl font-medium mb-4">Smart Glasses Integration</h3>
              <p className="text-text/70 leading-relaxed">
                Connects seamlessly with your smart glasses for an integrated viewing experience.
              </p>
            </div>
            
            <div className="bg-white rounded-2xl p-8 shadow-soft transition-all duration-300 hover:shadow-medium">
              <div className="h-12 w-12 bg-accent/20 rounded-xl flex items-center justify-center mb-6">
                <ShieldCheckIcon className="h-6 w-6 text-text" />
              </div>
              <h3 className="text-xl font-medium mb-4">Real-Time Object Recognition</h3>
              <p className="text-text/70 leading-relaxed">
                Transform your smart glasses into a tactile guide, converting real-world visuals into dynamic, intuitive feedback.              
              </p>
            </div>
            
            <div className="bg-white rounded-2xl p-8 shadow-soft transition-all duration-300 hover:shadow-medium">
              <div className="h-12 w-12 bg-accent/20 rounded-xl flex items-center justify-center mb-6">
                <BellAlertIcon className="h-6 w-6 text-text" />
              </div>
              <h3 className="text-xl font-medium mb-4">Directional Haptic Feedback</h3>
              <p className="text-text/70 leading-relaxed">
                Customize vibration patterns to effortlessly interpret your environment, empowering you with a natural, sixth sense.
              </p>
            </div>
          </div>
        </section>
        
        {/* Footer */}
        <footer className="pt-12 border-t border-text/10">
          <div className="flex flex-col md:flex-row justify-between items-center">
            <div className="flex items-center gap-2 mb-6 md:mb-0">
              <div className="h-8 w-8 bg-accent/20 rounded-full flex items-center justify-center">
                <Glasses size="sm" />
              </div>
              <span className="text-lg font-medium">SixthSense</span>
            </div>
            
            <div className="text-text/50 text-sm">
              © {new Date().getFullYear()} SixthSense. All rights reserved.
            </div>
          </div>
        </footer>
      </div>
    </div>
  );
}
