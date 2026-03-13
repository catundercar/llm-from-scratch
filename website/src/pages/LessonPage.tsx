import React, { useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { useLocale } from "../i18n";
import { getLesson, getPhaseContent, getLessonCount, getPhases } from "../data";
import type { ContentBlock, ContentSection, CodeExercise, LessonReference } from "../data";
import LanguageSwitcher from "../components/LanguageSwitcher";

const LessonPage: React.FC = () => {
  const { phaseId: pid, lessonId: lid } = useParams();
  const navigate = useNavigate();
  const { t, locale } = useLocale();

  const phaseId = Number(pid);
  const lessonId = Number(lid);
  const lesson = getLesson(phaseId, lessonId, locale);
  const phaseContent = getPhaseContent(phaseId, locale);
  const phases = getPhases(locale);
  const phase = phases.find((p) => p.id === phaseId);
  const totalLessons = getLessonCount(phaseId, locale);

  if (!lesson || !phaseContent || !phase) {
    return (
      <div
        style={{
          minHeight: "100vh",
          background: "#0A0A0B",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          gap: 16,
        }}
      >
        <div style={{ fontSize: 14, color: "#71717A" }}>
          {t("lesson.notFound")}
        </div>
        <button
          onClick={() => navigate("/")}
          style={{
            padding: "8px 20px",
            fontSize: 12,
            fontFamily: "inherit",
            background: "#18181B",
            border: "1px solid #27272A",
            borderRadius: 6,
            color: "#A1A1AA",
            cursor: "pointer",
          }}
        >
          {t("lesson.back")}
        </button>
      </div>
    );
  }

  const color = phaseContent.color;
  const accent = phaseContent.accent;
  const hasPrev = lessonId > 1;
  const hasNext = lessonId < totalLessons;

  return (
    <div style={{ minHeight: "100vh", background: "#0A0A0B" }}>
      {/* Header */}
      <header
        style={{
          borderBottom: "1px solid #18181B",
          padding: "20px 32px",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
        }}
      >
        <button
          onClick={() => navigate("/")}
          style={{
            fontSize: 12,
            fontFamily: "inherit",
            background: "none",
            border: "none",
            color: "#52525B",
            cursor: "pointer",
            padding: 0,
          }}
        >
          {t("lesson.back")}
        </button>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <a
            href="https://github.com/catundercar/llm-from-scratch"
            target="_blank"
            rel="noopener noreferrer"
            title="GitHub"
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              width: 32,
              height: 32,
              borderRadius: 6,
              border: "1px solid #27272A",
              background: "transparent",
              color: "#71717A",
              cursor: "pointer",
              transition: "all 0.15s",
              textDecoration: "none",
              flexShrink: 0,
            }}
            onMouseEnter={e => { (e.currentTarget as HTMLAnchorElement).style.borderColor = "#52525B"; (e.currentTarget as HTMLAnchorElement).style.color = "#E4E4E7"; }}
            onMouseLeave={e => { (e.currentTarget as HTMLAnchorElement).style.borderColor = "#27272A"; (e.currentTarget as HTMLAnchorElement).style.color = "#71717A"; }}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
              <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0 0 24 12c0-6.63-5.37-12-12-12z" />
            </svg>
          </a>
          <LanguageSwitcher />
        </div>
      </header>

      <main
        style={{
          maxWidth: 800,
          margin: "0 auto",
          padding: "32px 32px 64px",
        }}
      >
        {/* Phase badge + type/duration */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 10,
            marginBottom: 12,
          }}
        >
          <span
            style={{
              fontSize: 10,
              padding: "3px 10px",
              background: `${color}18`,
              border: `1px solid ${color}33`,
              borderRadius: 4,
              color: accent,
              letterSpacing: 1,
            }}
          >
            Phase {phaseId}
          </span>
          <span
            style={{
              fontSize: 10,
              padding: "3px 10px",
              background: "#18181B",
              border: "1px solid #27272A",
              borderRadius: 4,
              color: "#71717A",
            }}
          >
            {lesson.type} · {lesson.duration}
          </span>
        </div>

        {/* Title */}
        <h1
          style={{
            fontSize: 24,
            fontWeight: 700,
            color: "#E4E4E7",
            marginBottom: 6,
          }}
        >
          {lesson.title}
        </h1>
        <p
          style={{
            fontSize: 13,
            color: "#71717A",
            marginBottom: 24,
          }}
        >
          {lesson.subtitle}
        </p>

        {/* Prev/Next top navigation */}
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            marginBottom: 32,
          }}
        >
          {hasPrev ? (
            <button
              onClick={() => navigate(`/phase/${phaseId}/lesson/${lessonId - 1}`)}
              style={navBtnStyle}
            >
              {t("lesson.prev")}
            </button>
          ) : (
            <div />
          )}
          {hasNext ? (
            <button
              onClick={() => navigate(`/phase/${phaseId}/lesson/${lessonId + 1}`)}
              style={navBtnStyle}
            >
              {t("lesson.next")}
            </button>
          ) : (
            <div />
          )}
        </div>

        {/* Learning Objectives */}
        <SectionHeader label={t("lesson.objectives")} color={color} />
        <div
          style={{
            background: "#111113",
            border: "1px solid #1E1E22",
            borderRadius: 8,
            padding: "14px 18px",
            marginBottom: 32,
          }}
        >
          {lesson.objectives.map((obj, i) => (
            <div
              key={i}
              style={{
                fontSize: 12,
                color: "#A1A1AA",
                marginBottom: i < lesson.objectives.length - 1 ? 8 : 0,
                paddingLeft: 18,
                position: "relative",
                lineHeight: 1.5,
              }}
            >
              <span
                style={{
                  position: "absolute",
                  left: 0,
                  color: accent,
                  fontSize: 10,
                }}
              >
                ▸
              </span>
              {obj}
            </div>
          ))}
        </div>

        {/* Content Sections */}
        <SectionHeader label={t("lesson.content")} color={color} />
        <div style={{ marginBottom: 32 }}>
          {lesson.sections.map((section, i) => (
            <ExpandableSection
              key={i}
              section={section}
              color={color}
              accent={accent}
            />
          ))}
        </div>

        {/* Exercises */}
        {lesson.exercises.length > 0 && (
          <>
            <SectionHeader label={t("lesson.exercises")} color={color} />
            <div style={{ marginBottom: 32 }}>
              {lesson.exercises.map((ex) => (
                <ExerciseCard key={ex.id} exercise={ex} color={color} accent={accent} t={t} />
              ))}
            </div>
          </>
        )}

        {/* Acceptance Criteria */}
        {lesson.acceptanceCriteria.length > 0 && (
          <>
            <SectionHeader label={t("lesson.criteria")} color={color} />
            <div
              style={{
                background: "#111113",
                border: "1px solid #1E1E22",
                borderRadius: 8,
                padding: "14px 18px",
                marginBottom: 32,
              }}
            >
              {lesson.acceptanceCriteria.map((c, i) => (
                <div
                  key={i}
                  style={{
                    fontSize: 12,
                    color: "#A1A1AA",
                    marginBottom: i < lesson.acceptanceCriteria.length - 1 ? 6 : 0,
                    paddingLeft: 20,
                    position: "relative",
                    lineHeight: 1.5,
                  }}
                >
                  <span
                    style={{
                      position: "absolute",
                      left: 0,
                      color: accent,
                      fontSize: 11,
                    }}
                  >
                    ☐
                  </span>
                  {c}
                </div>
              ))}
            </div>
          </>
        )}

        {/* References */}
        {lesson.references.length > 0 && (
          <>
            <SectionHeader label={t("lesson.references")} color={color} />
            <div
              style={{
                display: "flex",
                flexDirection: "column",
                gap: 8,
                marginBottom: 32,
              }}
            >
              {lesson.references.map((ref, i) => (
                <ReferenceCard key={i} reference={ref} color={color} />
              ))}
            </div>
          </>
        )}

        {/* Bottom navigation */}
        <div
          style={{
            borderTop: "1px solid #1E1E22",
            paddingTop: 24,
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
          }}
        >
          {hasPrev ? (
            <button
              onClick={() => navigate(`/phase/${phaseId}/lesson/${lessonId - 1}`)}
              style={navBtnStyle}
            >
              {t("lesson.prev")}
            </button>
          ) : (
            <button
              onClick={() => navigate("/")}
              style={navBtnStyle}
            >
              {t("lesson.back")}
            </button>
          )}
          {hasNext ? (
            <button
              onClick={() => navigate(`/phase/${phaseId}/lesson/${lessonId + 1}`)}
              style={{
                ...navBtnStyle,
                background: `${color}22`,
                borderColor: `${color}44`,
                color: accent,
              }}
            >
              {t("lesson.next")}
            </button>
          ) : (
            <button
              onClick={() => navigate("/")}
              style={{
                ...navBtnStyle,
                background: `${color}22`,
                borderColor: `${color}44`,
                color: accent,
              }}
            >
              {t("lesson.complete", { phaseId })}
            </button>
          )}
        </div>
      </main>

      {/* Footer */}
      <footer
        style={{
          borderTop: "1px solid #18181B",
          padding: "24px 32px",
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
          gap: 6,
          fontSize: 12,
          color: "#52525B",
        }}
      >
        <span>Built with</span>
        <a
          href="https://github.com/catundercar/course-builder-plugin"
          target="_blank"
          rel="noopener noreferrer"
          style={{
            display: "inline-flex",
            alignItems: "center",
            gap: 4,
            color: "#71717A",
            textDecoration: "none",
            transition: "color 0.15s",
          }}
          onMouseEnter={e => (e.currentTarget as HTMLAnchorElement).style.color = "#E4E4E7"}
          onMouseLeave={e => (e.currentTarget as HTMLAnchorElement).style.color = "#71717A"}
        >
          <svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor">
            <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0 0 24 12c0-6.63-5.37-12-12-12z" />
          </svg>
          course-builder-plugin
        </a>
      </footer>
    </div>
  );
};

/* ─── Helpers ─── */

const navBtnStyle: React.CSSProperties = {
  padding: "8px 18px",
  fontSize: 12,
  fontFamily: "inherit",
  background: "#18181B",
  border: "1px solid #27272A",
  borderRadius: 6,
  color: "#A1A1AA",
  cursor: "pointer",
  transition: "all 0.15s",
};

const SectionHeader: React.FC<{ label: string; color: string }> = ({
  label,
  color,
}) => (
  <div
    style={{
      fontSize: 9,
      letterSpacing: 1.5,
      color: "#52525B",
      marginBottom: 10,
      display: "flex",
      alignItems: "center",
      gap: 8,
    }}
  >
    <span
      style={{
        width: 8,
        height: 8,
        borderRadius: 2,
        background: `${color}44`,
        display: "inline-block",
      }}
    />
    {label}
  </div>
);

/* ─── Expandable Content Section ─── */

const ExpandableSection: React.FC<{
  section: ContentSection;
  color: string;
  accent: string;
}> = ({ section, color, accent }) => {
  const [open, setOpen] = useState(false);

  return (
    <div
      style={{
        background: "#111113",
        border: "1px solid #1E1E22",
        borderRadius: 8,
        marginBottom: 6,
        overflow: "hidden",
      }}
    >
      <div
        onClick={() => setOpen(!open)}
        style={{
          padding: "12px 18px",
          cursor: "pointer",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
        }}
      >
        <span style={{ fontSize: 13, fontWeight: 500, color: "#E4E4E7" }}>
          {section.title}
        </span>
        <span
          style={{
            color: "#52525B",
            fontSize: 12,
            transition: "transform 0.2s",
            transform: open ? "rotate(90deg)" : "rotate(0deg)",
          }}
        >
          →
        </span>
      </div>
      {open && (
        <div
          style={{
            padding: "0 18px 16px",
            animation: "fadeIn 0.2s ease",
          }}
        >
          {section.blocks.map((block, i) => (
            <RenderBlock key={i} block={block} color={color} accent={accent} />
          ))}
        </div>
      )}
    </div>
  );
};

/* ─── Block Renderers ─── */

const RenderBlock: React.FC<{
  block: ContentBlock;
  color: string;
  accent: string;
}> = ({ block, color, accent }) => {
  switch (block.type) {
    case "paragraph":
      return (
        <p
          style={{
            fontSize: 12,
            color: "#A1A1AA",
            lineHeight: 1.7,
            marginBottom: 12,
            whiteSpace: "pre-wrap",
          }}
        >
          {block.text}
        </p>
      );
    case "heading":
      return (
        <div
          style={{
            fontSize: block.level === 2 ? 16 : block.level === 3 ? 14 : 12,
            fontWeight: 600,
            color: "#E4E4E7",
            marginTop: 16,
            marginBottom: 8,
          }}
        >
          {block.text}
        </div>
      );
    case "code":
      return (
        <pre
          style={{
            background: "#0A0A0B",
            border: "1px solid #27272A",
            borderRadius: 6,
            padding: "12px 16px",
            fontSize: 11,
            color: "#A1A1AA",
            overflow: "auto",
            marginBottom: 12,
            lineHeight: 1.6,
          }}
        >
          <code>{block.code}</code>
        </pre>
      );
    case "diagram":
      return (
        <pre
          style={{
            background: `${color}06`,
            border: `1px solid ${color}18`,
            borderRadius: 6,
            padding: "12px 16px",
            fontSize: 11,
            color: "#A1A1AA",
            overflow: "auto",
            marginBottom: 12,
            lineHeight: 1.6,
          }}
        >
          {block.content}
        </pre>
      );
    case "table":
      return (
        <div
          style={{
            overflow: "auto",
            marginBottom: 12,
            borderRadius: 6,
            border: "1px solid #27272A",
          }}
        >
          <table
            style={{
              width: "100%",
              borderCollapse: "collapse",
              fontSize: 11,
            }}
          >
            <thead>
              <tr>
                {block.headers.map((h, i) => (
                  <th
                    key={i}
                    style={{
                      textAlign: "left",
                      padding: "8px 12px",
                      background: "#18181B",
                      color: "#E4E4E7",
                      fontWeight: 600,
                      borderBottom: "1px solid #27272A",
                    }}
                  >
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {block.rows.map((row, ri) => (
                <tr key={ri}>
                  {row.map((cell, ci) => (
                    <td
                      key={ci}
                      style={{
                        padding: "6px 12px",
                        color: "#A1A1AA",
                        borderBottom: "1px solid #1E1E22",
                      }}
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
      const variants: Record<string, { bg: string; border: string; icon: string }> = {
        info: { bg: "#3B82F60A", border: "#3B82F633", icon: "ℹ" },
        warning: { bg: "#F59E0B0A", border: "#F59E0B33", icon: "⚠" },
        tip: { bg: "#10B9810A", border: "#10B98133", icon: "💡" },
        quote: { bg: "#8B5CF60A", border: "#8B5CF633", icon: "❝" },
      };
      const v = variants[block.variant] || variants.info;
      return (
        <div
          style={{
            background: v.bg,
            borderLeft: `2px solid ${v.border}`,
            borderRadius: "0 6px 6px 0",
            padding: "10px 14px",
            fontSize: 12,
            color: "#A1A1AA",
            lineHeight: 1.6,
            marginBottom: 12,
          }}
        >
          <span style={{ marginRight: 8 }}>{v.icon}</span>
          {block.text}
        </div>
      );
    }
    case "list":
      return (
        <div style={{ marginBottom: 12, paddingLeft: 16 }}>
          {block.items.map((item, i) => (
            <div
              key={i}
              style={{
                fontSize: 12,
                color: "#A1A1AA",
                lineHeight: 1.6,
                marginBottom: 4,
                position: "relative",
                paddingLeft: 14,
              }}
            >
              <span
                style={{
                  position: "absolute",
                  left: 0,
                  color: accent,
                  fontSize: 10,
                }}
              >
                {block.ordered ? `${i + 1}.` : "•"}
              </span>
              {item}
            </div>
          ))}
        </div>
      );
    default:
      return null;
  }
};

/* ─── Exercise Card ─── */

const ExerciseCard: React.FC<{
  exercise: CodeExercise;
  color: string;
  accent: string;
  t: (key: string) => string;
}> = ({ exercise, color, accent, t }) => {
  const [showPseudo, setShowPseudo] = useState(false);
  const [showHints, setShowHints] = useState(false);

  return (
    <div
      style={{
        background: "#111113",
        border: "1px solid #1E1E22",
        borderRadius: 8,
        padding: "14px 18px",
        marginBottom: 8,
      }}
    >
      <div
        style={{
          fontSize: 13,
          fontWeight: 600,
          color: "#E4E4E7",
          marginBottom: 4,
        }}
      >
        {exercise.title}
      </div>
      <div
        style={{
          fontSize: 12,
          color: "#71717A",
          marginBottom: 10,
          lineHeight: 1.5,
        }}
      >
        {exercise.description}
      </div>
      {exercise.labFile && (
        <div
          style={{
            fontSize: 10,
            color: "#52525B",
            marginBottom: 10,
            fontStyle: "italic",
          }}
        >
          📁 {exercise.labFile}
        </div>
      )}

      <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
        {exercise.pseudocode && (
          <button
            onClick={() => setShowPseudo(!showPseudo)}
            style={{
              fontSize: 10,
              fontFamily: "inherit",
              padding: "4px 12px",
              background: `${color}14`,
              border: `1px solid ${color}28`,
              borderRadius: 4,
              color: accent,
              cursor: "pointer",
            }}
          >
            {showPseudo ? t("lesson.hidePseudo") : t("lesson.showPseudo")}
          </button>
        )}
        {exercise.hints.length > 0 && (
          <button
            onClick={() => setShowHints(!showHints)}
            style={{
              fontSize: 10,
              fontFamily: "inherit",
              padding: "4px 12px",
              background: "#18181B",
              border: "1px solid #27272A",
              borderRadius: 4,
              color: "#71717A",
              cursor: "pointer",
            }}
          >
            {showHints ? t("lesson.hideHints") : t("lesson.showHints")}
          </button>
        )}
      </div>

      {showPseudo && exercise.pseudocode && (
        <pre
          style={{
            marginTop: 10,
            background: "#0A0A0B",
            border: "1px solid #27272A",
            borderRadius: 6,
            padding: "10px 14px",
            fontSize: 11,
            color: "#A1A1AA",
            overflow: "auto",
            lineHeight: 1.5,
            animation: "fadeIn 0.2s ease",
          }}
        >
          {exercise.pseudocode}
        </pre>
      )}

      {showHints && exercise.hints.length > 0 && (
        <div
          style={{
            marginTop: 10,
            padding: "10px 14px",
            background: `${color}06`,
            border: `1px solid ${color}14`,
            borderRadius: 6,
            animation: "fadeIn 0.2s ease",
          }}
        >
          {exercise.hints.map((hint, i) => (
            <div
              key={i}
              style={{
                fontSize: 11,
                color: "#A1A1AA",
                marginBottom: i < exercise.hints.length - 1 ? 4 : 0,
                paddingLeft: 14,
                position: "relative",
              }}
            >
              <span
                style={{ position: "absolute", left: 0, color: accent, fontSize: 10 }}
              >
                💡
              </span>
              {hint}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

/* ─── Reference Card ─── */

const ReferenceCard: React.FC<{
  reference: LessonReference;
  color: string;
}> = ({ reference, color }) => {
  return (
    <a
      href={reference.url}
      target="_blank"
      rel="noopener noreferrer"
      style={{
        display: "block",
        textDecoration: "none",
        background: "#111113",
        border: "1px solid #1E1E22",
        borderRadius: 8,
        padding: "12px 18px",
        transition: "border-color 0.15s",
      }}
      onMouseEnter={(e) =>
        (e.currentTarget.style.borderColor = `${color}44`)
      }
      onMouseLeave={(e) =>
        (e.currentTarget.style.borderColor = "#1E1E22")
      }
    >
      <div
        style={{
          fontSize: 12,
          fontWeight: 600,
          color: "#E4E4E7",
          marginBottom: 4,
        }}
      >
        {reference.title}
        <span
          style={{ fontSize: 10, color: "#52525B", marginLeft: 6 }}
        >
          ↗
        </span>
      </div>
      <div style={{ fontSize: 11, color: "#71717A", lineHeight: 1.5 }}>
        {reference.description}
      </div>
    </a>
  );
};

export default LessonPage;
