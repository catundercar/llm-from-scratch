"use client";

import { motion } from "framer-motion";

interface WhatsNewCardProps {
  newConcepts?: string[];
  newFunctions?: string[];
  locDelta?: number;
  coreAddition?: string;
  phaseColor?: string;
}

export function WhatsNewCard({
  newConcepts = [],
  newFunctions = [],
  locDelta = 0,
  coreAddition,
  phaseColor = "#3B82F6",
}: WhatsNewCardProps) {
  const hasContent = newConcepts.length > 0 || newFunctions.length > 0 || locDelta !== 0 || coreAddition;

  if (!hasContent) return null;

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold text-[var(--text-primary)]">What&apos;s New</h3>

      <div className="grid gap-4 sm:grid-cols-2">
        {coreAddition && (
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.05 }}
            className="rounded-xl border-2 p-4"
            style={{ borderColor: phaseColor, backgroundColor: `${phaseColor}10` }}
          >
            <h4 className="mb-1 text-xs font-medium text-[var(--text-secondary)] uppercase tracking-wider">
              Core Addition
            </h4>
            <p className="text-sm font-semibold" style={{ color: phaseColor }}>
              {coreAddition}
            </p>
          </motion.div>
        )}

        {newConcepts.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="rounded-xl border border-[var(--border)] bg-[var(--bg-secondary)] p-4"
          >
            <h4 className="mb-2 text-xs font-medium text-[var(--text-secondary)] uppercase tracking-wider">
              New Concepts
            </h4>
            <div className="flex flex-wrap gap-1.5">
              {newConcepts.map((concept) => (
                <span
                  key={concept}
                  className="rounded-full bg-emerald-500/10 px-3 py-1 font-mono text-xs font-medium text-emerald-700 dark:text-emerald-300"
                >
                  {concept}
                </span>
              ))}
            </div>
          </motion.div>
        )}

        {newFunctions.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.15 }}
            className="rounded-xl border border-[var(--border)] bg-[var(--bg-secondary)] p-4"
          >
            <h4 className="mb-2 text-xs font-medium text-[var(--text-secondary)] uppercase tracking-wider">
              New Functions
            </h4>
            <ul className="space-y-1 text-sm text-[var(--text-primary)]">
              {newFunctions.map((fn) => (
                <li key={fn} className="font-mono">
                  <span className="text-[var(--text-tertiary)]">def </span>
                  {fn}()
                </li>
              ))}
            </ul>
          </motion.div>
        )}

        {locDelta !== 0 && (
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="flex items-center rounded-xl border border-[var(--border)] bg-[var(--bg-secondary)] p-4"
          >
            <div>
              <h4 className="mb-1 text-xs font-medium text-[var(--text-secondary)] uppercase tracking-wider">
                Lines of Code
              </h4>
              <p className="text-2xl font-bold text-emerald-600 dark:text-emerald-400">
                {locDelta > 0 ? "+" : ""}{locDelta} LOC
              </p>
            </div>
          </motion.div>
        )}
      </div>
    </div>
  );
}
