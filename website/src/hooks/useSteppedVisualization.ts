"use client";
import { useState, useCallback, useEffect, useRef } from "react";

interface Options {
  totalSteps: number;
  autoPlayInterval?: number;
}

interface Return {
  currentStep: number;
  totalSteps: number;
  next: () => void;
  prev: () => void;
  reset: () => void;
  goToStep: (step: number) => void;
  isPlaying: boolean;
  toggleAutoPlay: () => void;
  isFirstStep: boolean;
  isLastStep: boolean;
}

export function useSteppedVisualization({
  totalSteps,
  autoPlayInterval = 2500,
}: Options): Return {
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const isFirstStep = currentStep === 0;
  const isLastStep = currentStep === totalSteps - 1;

  const next = useCallback(() => {
    setCurrentStep((prev) => (prev < totalSteps - 1 ? prev + 1 : prev));
  }, [totalSteps]);

  const prev = useCallback(() => {
    setCurrentStep((prev) => (prev > 0 ? prev - 1 : prev));
  }, []);

  const reset = useCallback(() => {
    setCurrentStep(0);
    setIsPlaying(false);
  }, []);

  const goToStep = useCallback(
    (step: number) => {
      if (step >= 0 && step < totalSteps) {
        setCurrentStep(step);
      }
    },
    [totalSteps]
  );

  const toggleAutoPlay = useCallback(() => {
    setIsPlaying((prev) => !prev);
  }, []);

  // Stop autoplay when reaching the last step
  useEffect(() => {
    if (isLastStep && isPlaying) {
      setIsPlaying(false);
    }
  }, [isLastStep, isPlaying]);

  // Autoplay interval
  useEffect(() => {
    if (isPlaying) {
      intervalRef.current = setInterval(() => {
        setCurrentStep((prev) => {
          if (prev >= totalSteps - 1) {
            return prev;
          }
          return prev + 1;
        });
      }, autoPlayInterval);
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [isPlaying, autoPlayInterval, totalSteps]);

  return {
    currentStep,
    totalSteps,
    next,
    prev,
    reset,
    goToStep,
    isPlaying,
    toggleAutoPlay,
    isFirstStep,
    isLastStep,
  };
}
