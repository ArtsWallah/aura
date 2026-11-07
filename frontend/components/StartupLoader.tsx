"use client";

import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
} from "react";

interface StartupContextType {
  isStartupReady: boolean;
  markStartupReady: () => void;
}

const StartupContext = createContext<StartupContextType>({
  isStartupReady: true,
  markStartupReady: () => {},
});

export function useStartup() {
  return useContext(StartupContext);
}

// In-memory module-level flag to ensure client-side ordinary page navigations never re-trigger the loader
let hasInitializedInSession = false;

interface StartupProviderProps {
  children: React.ReactNode;
}

export function StartupProvider({ children }: StartupProviderProps) {
  const [hasStarted, setHasStarted] = useState(() => {
    if (typeof window !== "undefined") {
      // If the page was refreshed (F5 / reload), reset the initialization flag so the loader plays once
      try {
        const navEntries = window.performance?.getEntriesByType?.("navigation");
        const isReload =
          (navEntries &&
            navEntries.length > 0 &&
            (navEntries[0] as PerformanceNavigationTiming).type === "reload") ||
          (window.performance as any)?.navigation?.type === 1;

        if (isReload) {
          sessionStorage.removeItem("aura_startup_initialized");
        }
      } catch {
        // Fallback for restricted storage environments
      }

      return (
        hasInitializedInSession ||
        Boolean(sessionStorage.getItem("aura_startup_initialized"))
      );
    }
    return false;
  });

  const [isAppReady, setIsAppReady] = useState(hasStarted);
  const [videoFinished, setVideoFinished] = useState(false);
  const [isDismissing, setIsDismissing] = useState(false);
  const [isMounted, setIsMounted] = useState(!hasStarted);
  const [playbackFailed, setPlaybackFailed] = useState(false);
  const [prefersReducedMotion, setPrefersReducedMotion] = useState(false);

  const videoRef = useRef<HTMLVideoElement | null>(null);

  // 1. Detect prefers-reduced-motion
  useEffect(() => {
    if (typeof window === "undefined" || hasStarted) return;
    const mediaQuery = window.matchMedia("(prefers-reduced-motion: reduce)");
    setPrefersReducedMotion(mediaQuery.matches);

    const handleChange = (e: MediaQueryListEvent) => {
      setPrefersReducedMotion(e.matches);
    };

    mediaQuery.addEventListener?.("change", handleChange);
    return () => mediaQuery.removeEventListener?.("change", handleChange);
  }, [hasStarted]);

  // 2. Fallback timeout to ensure the loader cannot get stuck indefinitely
  useEffect(() => {
    if (hasStarted) return;

    // Safety ceiling: dismiss after at most 8.5s (full video is ~6.0s)
    const safetyTimer = setTimeout(() => {
      setIsAppReady(true);
      setVideoFinished(true);
    }, 8500);

    // Fallback baseline for app readiness in case dashboard doesn't call markStartupReady
    const baselineAppTimer = setTimeout(() => {
      setIsAppReady(true);
    }, 3000);

    return () => {
      clearTimeout(safetyTimer);
      clearTimeout(baselineAppTimer);
    };
  }, [hasStarted]);

  // 3. Fallback for reduced motion or video playback error: auto-finish after 2s
  useEffect(() => {
    if (hasStarted) return;
    if (prefersReducedMotion || playbackFailed) {
      const fallbackTimer = setTimeout(() => {
        setIsAppReady(true);
        setVideoFinished(true);
      }, 2000);
      return () => clearTimeout(fallbackTimer);
    }
  }, [hasStarted, prefersReducedMotion, playbackFailed]);

  // 4. Autoplay enforcement & error handling for muted video
  useEffect(() => {
    if (hasStarted || prefersReducedMotion || !videoRef.current) return;

    const video = videoRef.current;
    video.muted = true;

    const playPromise = video.play();
    if (playPromise !== undefined) {
      playPromise.catch(() => {
        // Autoplay blocked or unsupported; gracefully fall back to static visual
        setPlaybackFailed(true);
      });
    }
  }, [hasStarted, prefersReducedMotion]);

  // 5. Trigger buttery-smooth fade-away when the video ends AND app is ready
  useEffect(() => {
    if (!videoFinished || !isAppReady || !isMounted || isDismissing) return;

    setIsDismissing(true);
    hasInitializedInSession = true;
    try {
      sessionStorage.setItem("aura_startup_initialized", "true");
    } catch {
      // Storage unavailable or disabled in sandbox/private mode
    }

    const dismissTimer = setTimeout(() => {
      setIsMounted(false);
      setHasStarted(true);
    }, 1200); // 1200ms matches smooth cinematic dissolve curve

    return () => clearTimeout(dismissTimer);
  }, [videoFinished, isAppReady, isMounted, isDismissing]);

  const markStartupReady = useCallback(() => {
    setIsAppReady(true);
  }, []);

  const handleVideoEnded = useCallback(() => {
    setVideoFinished(true);
  }, []);

  const handleTimeUpdate = useCallback(() => {
    // Initiate transition right as the video concludes its final motion (~0.4s before hard stop)
    // to achieve an organic, seamless dissolve into the dashboard without frame freezing.
    const video = videoRef.current;
    if (video && video.duration > 0 && video.currentTime >= video.duration - 0.4) {
      setVideoFinished(true);
    }
  }, []);

  return (
    <StartupContext.Provider value={{ isStartupReady: isAppReady, markStartupReady }}>
      {children}
      {isMounted && (
        <div
          role="status"
          aria-live="polite"
          aria-label="AURA is initializing"
          style={{
            transitionDuration: "1200ms",
            transitionTimingFunction: "cubic-bezier(0.22, 1, 0.36, 1)",
          }}
          className={`fixed inset-0 z-50 flex items-center justify-center bg-white select-none overflow-hidden transition-all ${
            isDismissing
              ? "opacity-0 scale-[1.015] pointer-events-none"
              : "opacity-100 scale-100"
          }`}
        >
          {/* Full-screen video asset extending edge-to-edge */}
          {!prefersReducedMotion && !playbackFailed ? (
            <video
              ref={videoRef}
              autoPlay
              muted
              playsInline
              preload="auto"
              controls={false}
              disablePictureInPicture
              disableRemotePlayback
              onEnded={handleVideoEnded}
              onTimeUpdate={handleTimeUpdate}
              onError={() => setPlaybackFailed(true)}
              className="absolute inset-0 w-full h-full object-cover border-0 outline-none bg-white will-change-transform"
            >
              <source src="/aura-startup.mp4" type="video/mp4" />
            </video>
          ) : (
            /* Static Fallback for reduced motion or video load error */
            <div
              className="flex items-center justify-center w-32 h-32 sm:w-40 sm:h-40"
              aria-hidden="true"
            >
              <svg
                viewBox="0 0 100 100"
                className="w-full h-full text-black"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
              >
                <circle
                  cx="50"
                  cy="50"
                  r="46"
                  stroke="currentColor"
                  strokeWidth="1.5"
                  strokeOpacity="0.15"
                />
                <circle
                  cx="50"
                  cy="50"
                  r="36"
                  stroke="currentColor"
                  strokeWidth="1.5"
                  strokeOpacity="0.3"
                  strokeDasharray="4 4"
                />
                <circle
                  cx="50"
                  cy="50"
                  r="26"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeOpacity="0.8"
                />
                <circle
                  cx="50"
                  cy="50"
                  r="16"
                  fill="currentColor"
                  fillOpacity="0.08"
                  stroke="currentColor"
                  strokeWidth="1.5"
                />
                <circle cx="50" cy="50" r="6" fill="currentColor" />
                <line
                  x1="50"
                  y1="2"
                  x2="50"
                  y2="8"
                  stroke="currentColor"
                  strokeWidth="2"
                />
                <line
                  x1="50"
                  y1="92"
                  x2="50"
                  y2="98"
                  stroke="currentColor"
                  strokeWidth="2"
                />
                <line
                  x1="2"
                  y1="50"
                  x2="8"
                  y2="50"
                  stroke="currentColor"
                  strokeWidth="2"
                />
                <line
                  x1="92"
                  y1="50"
                  x2="98"
                  y2="50"
                  stroke="currentColor"
                  strokeWidth="2"
                />
              </svg>
            </div>
          )}

          {/* Elegant initialization label floating above the bottom of the video */}
          <div
            className={`absolute bottom-10 sm:bottom-12 left-1/2 -translate-x-1/2 z-20 flex items-center justify-center gap-2.5 px-5 py-2.5 rounded-full bg-white/85 backdrop-blur-md border border-black/10 shadow-xs transition-all duration-700 ease-out ${
              isDismissing ? "opacity-0 translate-y-3" : "opacity-100 translate-y-0"
            }`}
          >
            <span className="h-1.5 w-1.5 rounded-full bg-black animate-pulse" />
            <span className="font-mono text-[11px] sm:text-xs font-bold uppercase tracking-[0.25em] text-zinc-700">
              INITIALIZING AURA...
            </span>
          </div>
        </div>
      )}
    </StartupContext.Provider>
  );
}
