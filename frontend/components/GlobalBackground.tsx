"use client";

import React, { useEffect, useRef, useState } from "react";

export default function GlobalBackground() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [prefersReducedMotion, setPrefersReducedMotion] = useState(false);
  const [playbackFailed, setPlaybackFailed] = useState(false);

  // 1. Detect prefers-reduced-motion
  useEffect(() => {
    if (typeof window === "undefined") return;
    const mediaQuery = window.matchMedia("(prefers-reduced-motion: reduce)");
    setPrefersReducedMotion(mediaQuery.matches);

    const handleChange = (e: MediaQueryListEvent) => {
      setPrefersReducedMotion(e.matches);
    };

    mediaQuery.addEventListener?.("change", handleChange);
    return () => mediaQuery.removeEventListener?.("change", handleChange);
  }, []);

  // 2. Autoplay enforcement & error handling for muted background video
  useEffect(() => {
    if (prefersReducedMotion || !videoRef.current) return;

    const video = videoRef.current;
    video.muted = true;

    const playPromise = video.play();
    if (playPromise !== undefined) {
      playPromise.catch(() => {
        // Autoplay blocked or unsupported; gracefully fall back to static visual
        setPlaybackFailed(true);
      });
    }
  }, [prefersReducedMotion]);

  return (
    <div
      aria-hidden="true"
      className="fixed inset-0 pointer-events-none z-0 overflow-hidden select-none bg-white"
    >
      {!prefersReducedMotion && !playbackFailed ? (
        <video
          ref={videoRef}
          autoPlay
          muted
          loop
          playsInline
          preload="auto"
          controls={false}
          disablePictureInPicture
          disableRemotePlayback
          tabIndex={-1}
          aria-hidden="true"
          className="absolute inset-0 w-full h-full object-cover border-0 outline-none opacity-20 transition-opacity duration-1000 ease-in-out"
          onError={() => setPlaybackFailed(true)}
        >
          <source src="/aura-background.mp4" type="video/mp4" />
        </video>
      ) : (
        /* Monochromatic static fallback for reduced motion or video error */
        <div className="absolute inset-0 bg-white" aria-hidden="true" />
      )}
    </div>
  );
}

