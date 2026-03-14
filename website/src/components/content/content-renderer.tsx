"use client";

import type { ContentBlock, ContentSection } from "@/data/types";

function RenderBlock({ block }: { block: ContentBlock }) {
  switch (block.type) {
    case "paragraph":
      return (
        <div className="text-sm leading-relaxed text-[var(--text-secondary)] whitespace-pre-line">
          {block.text}
        </div>
      );

    case "heading": {
      const Tag = `h${block.level}` as "h2" | "h3" | "h4";
      const sizeClass =
        block.level === 2
          ? "text-xl font-bold"
          : block.level === 3
          ? "text-lg font-semibold"
          : "text-base font-medium";
      return (
        <Tag className={`${sizeClass} text-[var(--text-primary)] mt-4 mb-2`}>
          {block.text}
        </Tag>
      );
    }

    case "code":
      return (
        <div className="rounded-lg border border-[var(--border)] overflow-hidden my-3">
          {block.language && (
            <div className="px-3 py-1.5 bg-[var(--bg-tertiary)] border-b border-[var(--border)]">
              <span className="text-xs font-mono text-[var(--text-tertiary)]">
                {block.language}
              </span>
            </div>
          )}
          <pre className="p-4 bg-zinc-950 overflow-x-auto">
            <code className="text-xs sm:text-sm font-mono text-zinc-200 leading-relaxed">
              {block.code}
            </code>
          </pre>
        </div>
      );

    case "diagram":
      return (
        <pre className="p-4 rounded-lg bg-[var(--bg-tertiary)] border border-[var(--border)] overflow-x-auto font-mono text-xs sm:text-sm text-[var(--text-primary)] leading-relaxed my-3">
          {block.content}
        </pre>
      );

    case "table":
      return (
        <div className="overflow-x-auto my-3 rounded-lg border border-[var(--border)]">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-[var(--bg-tertiary)]">
                {block.headers.map((h, i) => (
                  <th
                    key={i}
                    className="px-4 py-2 text-left font-semibold text-[var(--text-primary)] border-b border-[var(--border)]"
                  >
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {block.rows.map((row, ri) => (
                <tr
                  key={ri}
                  className="border-b border-[var(--border)] last:border-0"
                >
                  {row.map((cell, ci) => (
                    <td
                      key={ci}
                      className="px-4 py-2 text-[var(--text-secondary)]"
                    >
                      {cell}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      );

    case "callout": {
      const variantStyles: Record<string, { border: string; bg: string; text: string }> = {
        info: {
          border: "border-blue-400/30",
          bg: "bg-blue-500/10",
          text: "text-blue-300",
        },
        warning: {
          border: "border-amber-400/30",
          bg: "bg-amber-500/10",
          text: "text-amber-300",
        },
        tip: {
          border: "border-emerald-400/30",
          bg: "bg-emerald-500/10",
          text: "text-emerald-300",
        },
        quote: {
          border: "border-purple-400/30",
          bg: "bg-purple-500/10",
          text: "text-purple-300",
        },
      };
      const style = variantStyles[block.variant] ?? variantStyles.info;
      const icons: Record<string, string> = {
        info: "\u2139\uFE0F",
        warning: "\u26A0\uFE0F",
        tip: "\uD83D\uDCA1",
        quote: "\uD83D\uDCDD",
      };
      return (
        <div
          className={`rounded-lg border ${style.border} ${style.bg} p-4 my-3`}
        >
          <p className={`text-sm ${style.text} leading-relaxed`}>
            <span className="mr-2">{icons[block.variant]}</span>
            {block.text}
          </p>
        </div>
      );
    }

    case "list":
      return block.ordered ? (
        <ol className="list-decimal list-inside space-y-1 my-2 text-sm text-[var(--text-secondary)]">
          {block.items.map((item, i) => (
            <li key={i}>{item}</li>
          ))}
        </ol>
      ) : (
        <ul className="list-disc list-inside space-y-1 my-2 text-sm text-[var(--text-secondary)]">
          {block.items.map((item, i) => (
            <li key={i}>{item}</li>
          ))}
        </ul>
      );

    default:
      return null;
  }
}

export function ContentRenderer({ blocks }: { blocks: ContentBlock[] }) {
  return (
    <div className="space-y-3">
      {blocks.map((block, i) => (
        <RenderBlock key={i} block={block} />
      ))}
    </div>
  );
}

export function SectionRenderer({ section }: { section: ContentSection }) {
  return (
    <div className="space-y-3">
      <h3 className="text-lg font-semibold text-[var(--text-primary)]">
        {section.title}
      </h3>
      <ContentRenderer blocks={section.blocks} />
    </div>
  );
}
