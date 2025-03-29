"use client";

import { useState } from "react";
import Link from "next/link";
import { ArrowRightIcon, VideoCameraIcon, ShieldCheckIcon, BellAlertIcon, EyeIcon } from "@heroicons/react/24/outline";

export default function Home() {
  return (
    <div className="bg-background min-h-screen text-text">
      <div className="max-w-4xl mx-auto px-6 py-24">
        {/* Header */}
        <header className="mb-24">
          <div className="flex items-center gap-3">
            <div className="h-10 w-10 bg-accent/20 rounded-full flex items-center justify-center shadow-soft">
              <VideoCameraIcon className="h-5 w-5 text-text" />
            </div>
            <h1 className="text-2xl font-medium">FeedFeel</h1>
          </div>
        </header>

        {/* Hero Section */}
        <section className="mb-32">
          <h2 className="text-4xl md:text-5xl font-medium leading-tight mb-8 max-w-2xl">
            See through your smart glasses with elegant simplicity
          </h2>
          
          <p className="text-base leading-relaxed mb-12 max-w-2xl text-text/80">
            FeedFeel seamlessly integrates with your smart glasses to deliver a thoughtful viewing experience.
            View your camera feed with precision while keeping important information at your fingertips.
          </p>
          
          <div className="flex flex-col sm:flex-row gap-6">
            <Link
              href="/dashboard"
              className="group flex items-center justify-center rounded-xl bg-accent px-8 py-3.5 text-text font-medium transition-all duration-300 shadow-soft hover:shadow-medium"
            >
              <span>Go to Dashboard</span>
              <ArrowRightIcon className="ml-2 h-5 w-5 group-hover:translate-x-1 transition-transform" />
            </Link>
            <a
              href="#features"
              className="rounded-xl px-8 py-3.5 text-text/70 font-medium bg-background border border-text/10 hover:border-text/20 transition-all duration-300"
            >
              Learn More
            </a>
          </div>
        </section>

        {/* Smart Glasses Visual */}
        <section className="mb-32">
          <div className="relative w-full max-w-2xl mx-auto p-8 bg-white rounded-2xl shadow-soft overflow-hidden">
            <div className="absolute top-0 right-0 h-32 w-32 bg-accent/10 rounded-full -mr-16 -mt-16 z-0"></div>
            
            {/* Glasses illustration */}
            <div className="relative z-10 flex justify-center py-8">
              <div className="relative">
                {/* Glasses frame */}
                <div className="w-[300px] h-[100px] relative">
                  {/* Left lens */}
                  <div className="absolute left-6 top-6 w-[100px] h-[60px] rounded-full bg-background border border-text/10 shadow-soft overflow-hidden">
                    <div className="absolute inset-0 flex items-center justify-center">
                      <EyeIcon className="h-6 w-6 text-text/30" />
                    </div>
                  </div>
                  
                  {/* Right lens */}
                  <div className="absolute right-6 top-6 w-[100px] h-[60px] rounded-full bg-background border border-text/10 shadow-soft overflow-hidden">
                    <div className="absolute inset-0 flex items-center justify-center">
                      <VideoCameraIcon className="h-6 w-6 text-accent-dark/70" />
                    </div>
                  </div>
                  
                  {/* Bridge */}
                  <div className="absolute left-1/2 top-[36px] transform -translate-x-1/2 w-[36px] h-[12px] bg-background border border-text/10 rounded-full"></div>
                  
                  {/* Temple arms */}
                  <div className="absolute left-2 top-[36px] w-[24px] h-[4px] bg-background border-t border-text/10"></div>
                  <div className="absolute right-2 top-[36px] w-[24px] h-[4px] bg-background border-t border-text/10"></div>
                </div>
                
                {/* Label */}
                <div className="text-center mt-8">
                  <p className="text-sm text-text/60">Smart Camera Technology</p>
                </div>
              </div>
            </div>
            
            {/* Caption */}
            <div className="text-center mt-4">
              <p className="text-text/70 text-base leading-relaxed">
                View your world through an elegant interface
              </p>
            </div>
          </div>
        </section>

        {/* Features */}
        <section id="features" className="mb-32">
          <h2 className="text-2xl font-medium mb-16 text-center">Key Features</h2>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-12">
            <div className="bg-white rounded-2xl p-8 shadow-soft transition-all duration-300 hover:shadow-medium">
              <div className="h-12 w-12 bg-accent/20 rounded-xl flex items-center justify-center mb-6">
                <VideoCameraIcon className="h-6 w-6 text-text" />
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
              <h3 className="text-xl font-medium mb-4">Smart Fallback</h3>
              <p className="text-text/70 leading-relaxed">
                Gracefully falls back to your system camera when needed, ensuring continuity.
              </p>
            </div>
            
            <div className="bg-white rounded-2xl p-8 shadow-soft transition-all duration-300 hover:shadow-medium">
              <div className="h-12 w-12 bg-accent/20 rounded-xl flex items-center justify-center mb-6">
                <BellAlertIcon className="h-6 w-6 text-text" />
              </div>
              <h3 className="text-xl font-medium mb-4">Information Dashboard</h3>
              <p className="text-text/70 leading-relaxed">
                View important information alongside your camera feed without distraction.
              </p>
            </div>
          </div>
        </section>
        
        {/* Footer */}
        <footer className="pt-12 border-t border-text/10">
          <div className="flex flex-col md:flex-row justify-between items-center">
            <div className="flex items-center gap-2 mb-6 md:mb-0">
              <div className="h-8 w-8 bg-accent/20 rounded-full flex items-center justify-center">
                <VideoCameraIcon className="h-4 w-4 text-text" />
              </div>
              <span className="text-lg font-medium">FeedFeel</span>
            </div>
            
            <div className="text-text/50 text-sm">
              © {new Date().getFullYear()} FeedFeel. All rights reserved.
            </div>
          </div>
        </footer>
      </div>
    </div>
  );
}
