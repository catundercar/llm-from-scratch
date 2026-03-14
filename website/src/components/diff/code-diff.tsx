"use client";

import { useState, useMemo } from "react";
import { diffLines, type Change } from "diff";

interface CodeDiffProps {
  oldSource: string;
  newSource: string;
  oldLabel: string;
  newLabel: string;
}

export function CodeDiff({ oldSource, newSource, oldLabel, newLabel }: CodeDiffProps) {
  const [viewMode, setViewMode] = useState<"unified" | "split">("unified");
  const changes = useMemo(() => diffLines(oldSource, newSource), [oldSource, newSource]);

  return (
    <div>
      <div className="mb-4 flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
        <div className="min-w-0 truncate text-sm text-[var(--text-secondary)]">
          <span className="font-medium text-[var(--text-primary)]">{oldLabel}</span>
          {" → "}
          <span className="font-medium text-[var(--text-primary)]">{newLabel}</span>
        </div>
        <div className="flex shrink-0 rounded-lg border border-[var(--border)] overflow-hidden">
          <button
            onClick={() => setViewMode("unified")}
            className={`min-h-[36px] px-3 text-xs font-medium transition-colors ${
              viewMode === "unified"
                ? "bg-[var(--text-primary)] text-[var(--bg-primary)]"
                : "text-[var(--text-secondary)] hover:text-[var(--text-primary)]"
            }`}
          >
            Unified
          </button>
          <button
            onClick={() => setViewMode("split")}
            className={`hidden sm:inline-flex min-h-[36px] px-3 text-xs font-medium transition-colors ${
              viewMode === "split"
                ? "bg-[var(--text-primary)] text-[var(--bg-primary)]"
                : "text-[var(--text-secondary)] hover:text-[var(--text-primary)]"
            }`}
          >
            Split
          </button>
        </div>
      </div>

      {viewMode === "unified" ? (
        <UnifiedView changes={changes} />
      ) : (
        <SplitView changes={changes} />
      )}
    </div>
  );
}

function UnifiedView({ changes }: { changes: Change[] }) {
  let oldLine = 1;
  let newLine = 1;

  const rows: { oldNum: number | null; newNum: number | null; type: "add" | "remove" | "context"; text: string }[] = [];

  for (const change of changes) {
    const lines = change.value.replace(/\n$/, "").split("\n");
    for (const line of lines) {
      if (change.added) {
        rows.push({ oldNum: null, newNum: newLine++, type: "add", text: line });
      } else if (change.removed) {
        rows.push({ oldNum: oldLine++, newNum: null, type: "remove", text: line });
      } else {
        rows.push({ oldNum: oldLine++, newNum: newLine++, type: "context", text: line });
      }
    }
  }

  return (
    <div className="overflow-x-auto rounded-xl border border-[var(--border)]">
      <table className="w-full border-collapse font-mono text-xs leading-5">
        <tbody>
          {rows.map((row, i) => (
            <tr
              key={i}
              className={
                row.type === "add"
                  ? "bg-emerald-500/10"
                  : row.type === "remove"
                    ? "bg-red-500/10"
                    : ""
              }
            >
              <td className="w-10 select-none border-r border-[var(--border)] px-2 text-right text-[var(--text-tertiary)]">
                {row.oldNum ?? ""}
              </td>
              <td className="w-10 select-none border-r border-[var(--border)] px-2 text-right text-[var(--text-tertiary)]">
                {row.newNum ?? ""}
              </td>
              <td className="w-4 select-none px-1 text-center">
                {row.type === "add" && <span className="text-emerald-500">+</span>}
                {row.type === "remove" && <span className="text-red-500">-</span>}
              </td>
              <td className="whitespace-pre px-2">
                <span
                  className={
                    row.type === "add"
                      ? "text-emerald-700 dark:text-emerald-300"
                      : row.type === "remove"
                        ? "text-red-700 dark:text-red-300"
                        : "text-[var(--text-primary)]"
                  }
                >
                  {row.text}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function SplitView({ changes }: { changes: Change[] }) {
  let oldLine = 1;
  let newLine = 1;

  type SplitRow = {
    left: { num: number | null; text: string; type: "remove" | "context" | "empty" };
    right: { num: number | null; text: string; type: "add" | "context" | "empty" };
  };

  const rows: SplitRow[] = [];

  for (const change of changes) {
    const lines = change.value.replace(/\n$/, "").split("\n");
    if (change.removed) {
      for (const line of lines) {
        rows.push({
          left: { num: oldLine++, text: line, type: "remove" },
          right: { num: null, text: "", type: "empty" },
        });
      }
    } else if (change.added) {
      let filled = 0;
      for (const line of lines) {
        const lastUnfilled = rows.length - lines.length + filled;
        if (
          lastUnfilled >= 0 &&
          lastUnfilled < rows.length &&
          rows[lastUnfilled].right.type === "empty" &&
          rows[lastUnfilled].left.type === "remove"
        ) {
          rows[lastUnfilled].right = { num: newLine++, text: line, type: "add" };
        } else {
          rows.push({
            left: { num: null, text: "", type: "empty" },
            right: { num: newLine++, text: line, type: "add" },
          });
        }
        filled++;
      }
    } else {
      for (const line of lines) {
        rows.push({
          left: { num: oldLine++, text: line, type: "context" },
          right: { num: newLine++, text: line, type: "context" },
        });
      }
    }
  }

  const cellBg = (type: string) =>
    type === "add"
      ? "bg-emerald-500/10"
      : type === "remove"
        ? "bg-red-500/10"
        : type === "empty"
          ? "bg-[var(--bg-tertiary)]/30"
          : "";

  const cellText = (type: string) =>
    type === "add"
      ? "text-emerald-700 dark:text-emerald-300"
      : type === "remove"
        ? "text-red-700 dark:text-red-300"
        : "text-[var(--text-primary)]";

  return (
    <div className="overflow-x-auto rounded-xl border border-[var(--border)]">
      <table className="w-full border-collapse font-mono text-xs leading-5">
        <tbody>
          {rows.map((row, i) => (
            <tr key={i}>
              <td className="w-10 select-none border-r border-[var(--border)] px-2 text-right text-[var(--text-tertiary)]">
                {row.left.num ?? ""}
              </td>
              <td className={`w-1/2 border-r border-[var(--border)] whitespace-pre px-2 ${cellBg(row.left.type)} ${cellText(row.left.type)}`}>
                {row.left.text}
              </td>
              <td className="w-10 select-none border-r border-[var(--border)] px-2 text-right text-[var(--text-tertiary)]">
                {row.right.num ?? ""}
              </td>
              <td className={`w-1/2 whitespace-pre px-2 ${cellBg(row.right.type)} ${cellText(row.right.type)}`}>
                {row.right.text}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
