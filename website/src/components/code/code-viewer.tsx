"use client";

import { useMemo } from "react";

interface CodeViewerProps {
  source: string;
  filename: string;
  language?: string;
}

const PYTHON_KEYWORDS = new Set([
  "def", "class", "import", "from", "return", "if", "elif", "else",
  "while", "for", "in", "not", "and", "or", "is", "None", "True",
  "False", "try", "except", "raise", "with", "as", "yield", "break",
  "continue", "pass", "global", "lambda", "async", "await", "self",
]);

function highlightPythonLine(line: string): React.ReactNode[] {
  const trimmed = line.trimStart();

  if (trimmed.startsWith("#")) {
    return [<span key={0} className="text-zinc-500 dark:text-zinc-500 italic">{line}</span>];
  }

  if (trimmed.startsWith("@")) {
    return [<span key={0} className="text-amber-600 dark:text-amber-400">{line}</span>];
  }

  if (trimmed.startsWith('"""') || trimmed.startsWith("'''")) {
    return [<span key={0} className="text-emerald-600 dark:text-emerald-400">{line}</span>];
  }

  const parts = line.split(
    /(\b(?:def|class|import|from|return|if|elif|else|while|for|in|not|and|or|is|None|True|False|try|except|raise|with|as|yield|break|continue|pass|global|lambda|async|await|self)\b|"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|f"(?:[^"\\]|\\.)*"|f'(?:[^'\\]|\\.)*'|#.*$|\b\d+(?:\.\d+)?\b)/
  );

  return parts.map((part, idx) => {
    if (!part) return null;
    if (part === "self") {
      return <span key={idx} className="text-purple-600 dark:text-purple-400">{part}</span>;
    }
    if (PYTHON_KEYWORDS.has(part)) {
      return <span key={idx} className="text-blue-600 dark:text-blue-400 font-medium">{part}</span>;
    }
    if (part.startsWith("#")) {
      return <span key={idx} className="text-zinc-500 dark:text-zinc-500 italic">{part}</span>;
    }
    if (
      (part.startsWith('"') && part.endsWith('"')) ||
      (part.startsWith("'") && part.endsWith("'")) ||
      (part.startsWith('f"') && part.endsWith('"')) ||
      (part.startsWith("f'") && part.endsWith("'"))
    ) {
      return <span key={idx} className="text-emerald-600 dark:text-emerald-400">{part}</span>;
    }
    if (/^\d+(?:\.\d+)?$/.test(part)) {
      return <span key={idx} className="text-orange-600 dark:text-orange-400">{part}</span>;
    }
    return <span key={idx}>{part}</span>;
  });
}

export function CodeViewer({ source, filename }: CodeViewerProps) {
  const lines = useMemo(() => source.split("\n"), [source]);

  return (
    <div className="rounded-xl border border-[var(--border)] overflow-hidden">
      {/* Terminal-style header */}
      <div className="flex items-center gap-2 border-b border-[var(--border)] px-4 py-2.5 bg-[var(--bg-tertiary)]">
        <div className="flex gap-1.5">
          <span className="h-3 w-3 rounded-full bg-red-500/80" />
          <span className="h-3 w-3 rounded-full bg-yellow-500/80" />
          <span className="h-3 w-3 rounded-full bg-green-500/80" />
        </div>
        <span className="font-mono text-xs text-[var(--text-tertiary)] ml-2">{filename}</span>
      </div>

      {/* Code content */}
      <div className="overflow-x-auto bg-zinc-950 dark:bg-zinc-950">
        <pre className="p-4 text-xs leading-5 sm:text-sm sm:leading-6">
          <code>
            {lines.map((line, i) => (
              <div key={i} className="flex hover:bg-zinc-900/50">
                <span className="mr-4 inline-block w-8 shrink-0 select-none text-right text-zinc-600 font-mono">
                  {i + 1}
                </span>
                <span className="text-zinc-200">
                  {highlightPythonLine(line)}
                </span>
              </div>
            ))}
          </code>
        </pre>
      </div>
    </div>
  );
}
