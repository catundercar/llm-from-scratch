"use client";

import { Github } from "lucide-react";

export function Footer() {
  return (
    <footer className="border-t border-[var(--border)] bg-[var(--bg-primary)]">
      <div className="max-w-7xl mx-auto px-4 py-6 flex flex-col sm:flex-row items-center justify-between gap-4">
        <div className="flex items-center gap-4 text-sm text-[var(--text-secondary)]">
          <span className="font-semibold text-[var(--text-primary)]">
            LLM From Scratch
          </span>
          <span className="hidden sm:inline">|</span>
          <span>MIT License</span>
        </div>
        <a
          href="https://github.com/llm-from-scratch"
          target="_blank"
          rel="noopener noreferrer"
          className="text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors"
        >
          <Github size={18} />
        </a>
      </div>
    </footer>
  );
}
