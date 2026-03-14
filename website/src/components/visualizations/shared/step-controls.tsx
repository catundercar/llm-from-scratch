"use client";
import { Play, Pause, SkipBack, SkipForward, RotateCcw } from "lucide-react";

interface StepControlsProps {
  currentStep: number;
  totalSteps: number;
  onPrev: () => void;
  onNext: () => void;
  onReset: () => void;
  isPlaying: boolean;
  onToggleAutoPlay: () => void;
  stepTitle: string;
  stepDescription: string;
}

export function StepControls({
  currentStep,
  totalSteps,
  onPrev,
  onNext,
  onReset,
  isPlaying,
  onToggleAutoPlay,
  stepTitle,
  stepDescription,
}: StepControlsProps) {
  const isFirstStep = currentStep === 0;
  const isLastStep = currentStep === totalSteps - 1;

  return (
    <div className="flex flex-col gap-3">
      {/* Step annotation box */}
      <div
        className="rounded-lg border border-blue-200 bg-blue-50 px-4 py-3 dark:border-blue-900 dark:bg-blue-950/40"
      >
        <p
          className="text-sm font-semibold text-blue-900 dark:text-blue-200"
          style={{ fontFamily: "var(--font-geist-mono), monospace" }}
        >
          {stepTitle}
        </p>
        <p className="mt-1 text-sm text-blue-700 dark:text-blue-300">
          {stepDescription}
        </p>
      </div>

      {/* Control buttons row */}
      <div className="flex items-center justify-center gap-2">
        <button
          onClick={onReset}
          disabled={isFirstStep && !isPlaying}
          className="rounded-lg p-2 text-[var(--foreground)] transition-colors hover:bg-zinc-200 disabled:opacity-30 disabled:cursor-not-allowed dark:hover:bg-zinc-800"
          aria-label="Reset"
        >
          <RotateCcw size={16} />
        </button>

        <button
          onClick={onPrev}
          disabled={isFirstStep}
          className="rounded-lg p-2 text-[var(--foreground)] transition-colors hover:bg-zinc-200 disabled:opacity-30 disabled:cursor-not-allowed dark:hover:bg-zinc-800"
          aria-label="Previous step"
        >
          <SkipBack size={16} />
        </button>

        <button
          onClick={onToggleAutoPlay}
          disabled={isLastStep && !isPlaying}
          className="rounded-full bg-zinc-900 p-2.5 text-white transition-colors hover:bg-zinc-700 disabled:opacity-30 disabled:cursor-not-allowed dark:bg-zinc-100 dark:text-zinc-900 dark:hover:bg-zinc-300"
          aria-label={isPlaying ? "Pause" : "Play"}
        >
          {isPlaying ? <Pause size={16} /> : <Play size={16} />}
        </button>

        <button
          onClick={onNext}
          disabled={isLastStep}
          className="rounded-lg p-2 text-[var(--foreground)] transition-colors hover:bg-zinc-200 disabled:opacity-30 disabled:cursor-not-allowed dark:hover:bg-zinc-800"
          aria-label="Next step"
        >
          <SkipForward size={16} />
        </button>
      </div>

      {/* Step indicator dots + counter */}
      <div className="flex items-center justify-center gap-3">
        <div className="flex items-center gap-1.5">
          {Array.from({ length: totalSteps }, (_, i) => (
            <span
              key={i}
              className={`inline-block h-2 w-2 rounded-full transition-colors ${
                i <= currentStep
                  ? "bg-zinc-900 dark:bg-zinc-100"
                  : "bg-zinc-300 dark:bg-zinc-700"
              }`}
            />
          ))}
        </div>
        <span
          className="text-xs text-zinc-500 dark:text-zinc-400"
          style={{ fontFamily: "var(--font-geist-mono), monospace" }}
        >
          {currentStep + 1}/{totalSteps}
        </span>
      </div>
    </div>
  );
}
