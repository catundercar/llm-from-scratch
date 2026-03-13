import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useLocale } from "../i18n";
import { getPhases, architecture, getPrinciples, dataFlowSteps, hasLessons } from "../data";
import type { Phase, ArchitectureLayer, Principle } from "../data";
import LanguageSwitcher from "./LanguageSwitcher";

type Tab = "roadmap" | "architecture" | "principles";

const CourseRoadmap: React.FC = () => {
  const { t, locale } = useLocale();
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState<Tab>("roadmap");
  const [expandedPhase, setExpandedPhase] = useState<number | null>(null);

  const phases = getPhases(locale);
  const principles = getPrinciples(locale);

  const tabs: { key: Tab; label: string }[] = [
    { key: "roadmap", label: t("tab.roadmap") },
    { key: "architecture", label: t("tab.architecture") },
    { key: "principles", label: t("tab.principles") },
  ];

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
        <div>
          <div
            style={{
              fontSize: 10,
              letterSpacing: 2,
              color: "#52525B",
              marginBottom: 4,
              textTransform: "uppercase",
            }}
          >
            {t("header.badge")}
          </div>
          <div style={{ fontSize: 16, fontWeight: 700, color: "#E4E4E7" }}>
            {t("header.title")}
          </div>
        </div>
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

      {/* Hero */}
      <div
        style={{
          maxWidth: 960,
          margin: "0 auto",
          padding: "48px 32px 0",
        }}
      >
        <h1
          style={{
            fontSize: 28,
            fontWeight: 700,
            color: "#E4E4E7",
            lineHeight: 1.3,
            marginBottom: 8,
          }}
        >
          {t("header.subtitle1")}
        </h1>
        <p style={{ fontSize: 13, color: "#71717A", marginBottom: 32 }}>
          {t("header.subtitle2")}
        </p>

        {/* Tabs */}
        <div
          style={{
            display: "flex",
            gap: 0,
            borderBottom: "1px solid #27272A",
            marginBottom: 32,
          }}
        >
          {tabs.map((tab) => (
            <button
              key={tab.key}
              onClick={() => setActiveTab(tab.key)}
              style={{
                padding: "10px 20px",
                fontSize: 12,
                fontFamily: "inherit",
                background: "transparent",
                border: "none",
                borderBottom:
                  activeTab === tab.key
                    ? "2px solid #E4E4E7"
                    : "2px solid transparent",
                color: activeTab === tab.key ? "#E4E4E7" : "#52525B",
                cursor: "pointer",
                transition: "all 0.15s",
              }}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      {/* Content */}
      <main style={{ maxWidth: 960, margin: "0 auto", padding: "0 32px 64px" }}>
        {activeTab === "roadmap" && (
          <RoadmapTab
            phases={phases}
            expandedPhase={expandedPhase}
            onToggle={(id) =>
              setExpandedPhase(expandedPhase === id ? null : id)
            }
            onEnterLesson={(phaseId) =>
              navigate(`/phase/${phaseId}/lesson/1`)
            }
            t={t}
            locale={locale}
          />
        )}
        {activeTab === "architecture" && <ArchitectureTab t={t} />}
        {activeTab === "principles" && (
          <PrinciplesTab principles={principles} t={t} />
        )}
      </main>

      {/* Footer */}
      <footer
        style={{
          borderTop: "1px solid #18181B",
          padding: "24px 32px",
          display: "flex",
          flexDirection: "column",
          justifyContent: "center",
          alignItems: "center",
          gap: 12,
          fontSize: 12,
          color: "#52525B",
        }}
      >
        <img
          src="https://hits.sh/llm-from-scratch.catuc.club.svg?style=flat-square&color=27272A&labelColor=18181B&label=visitors"
          alt="visitor count"
          style={{ height: 18, borderRadius: 3 }}
        />
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
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
        </div>
      </footer>
    </div>
  );
};

/* ─── Roadmap Tab ─── */

const RoadmapTab: React.FC<{
  phases: Phase[];
  expandedPhase: number | null;
  onToggle: (id: number) => void;
  onEnterLesson: (phaseId: number) => void;
  t: (key: string) => string;
  locale: import("../i18n/LocaleContext").Locale;
}> = ({ phases, expandedPhase, onToggle, onEnterLesson, t, locale }) => {
  return (
    <div style={{ position: "relative" }}>
      {phases.map((phase, idx) => {
        const isExpanded = expandedPhase === phase.id;
        const isLast = idx === phases.length - 1;
        return (
          <div
            key={phase.id}
            style={{ position: "relative", paddingLeft: 56, marginBottom: 8 }}
          >
            {/* Connector line */}
            {!isLast && (
              <div
                style={{
                  position: "absolute",
                  left: 19,
                  top: 44,
                  bottom: -2,
                  width: 1,
                  background: `${phase.color}33`,
                }}
              />
            )}
            {/* Icon dot */}
            <div
              style={{
                position: "absolute",
                left: 0,
                top: 2,
                width: 40,
                height: 40,
                borderRadius: "50%",
                background: `${phase.color}18`,
                border: `1px solid ${phase.color}44`,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                fontSize: 18,
              }}
            >
              {phase.icon}
            </div>

            {/* Phase card */}
            <div
              onClick={() => onToggle(phase.id)}
              style={{
                background: "#111113",
                border: "1px solid #1E1E22",
                borderRadius: 8,
                padding: "14px 18px",
                cursor: "pointer",
                transition: "border-color 0.15s",
              }}
              onMouseEnter={(e) =>
                (e.currentTarget.style.borderColor = `${phase.color}44`)
              }
              onMouseLeave={(e) =>
                (e.currentTarget.style.borderColor = "#1E1E22")
              }
            >
              {/* Header row */}
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "flex-start",
                }}
              >
                <div style={{ flex: 1 }}>
                  <div
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: 10,
                      marginBottom: 4,
                    }}
                  >
                    <span
                      style={{
                        fontSize: 10,
                        color: phase.accent,
                        letterSpacing: 1,
                      }}
                    >
                      {phase.week}
                    </span>
                    <span style={{ fontSize: 10, color: "#52525B" }}>
                      {phase.duration}
                    </span>
                  </div>
                  <div
                    style={{
                      fontSize: 14,
                      fontWeight: 600,
                      color: "#E4E4E7",
                      marginBottom: 2,
                    }}
                  >
                    {phase.title}
                  </div>
                  <div style={{ fontSize: 11, color: "#52525B" }}>
                    {phase.subtitle}
                  </div>
                </div>
                <span
                  style={{
                    color: "#52525B",
                    fontSize: 14,
                    transition: "transform 0.2s",
                    transform: isExpanded ? "rotate(90deg)" : "rotate(0deg)",
                    marginTop: 4,
                  }}
                >
                  →
                </span>
              </div>

              {/* Expanded content */}
              {isExpanded && (
                <div
                  style={{
                    marginTop: 16,
                    animation: "fadeIn 0.2s ease",
                  }}
                >
                  {/* Goal block */}
                  <div
                    style={{
                      borderLeft: `2px solid ${phase.color}66`,
                      background: `${phase.color}08`,
                      padding: "10px 14px",
                      borderRadius: "0 6px 6px 0",
                      fontSize: 12,
                      color: "#A1A1AA",
                      lineHeight: 1.6,
                      marginBottom: 16,
                    }}
                  >
                    {phase.goal}
                  </div>

                  {/* Concepts */}
                  <div style={{ marginBottom: 14 }}>
                    <div
                      style={{
                        fontSize: 9,
                        letterSpacing: 1.5,
                        color: "#52525B",
                        marginBottom: 8,
                      }}
                    >
                      {t("phase.concepts")}
                    </div>
                    <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                      {phase.concepts.map((c, i) => (
                        <span
                          key={i}
                          style={{
                            fontSize: 11,
                            padding: "3px 10px",
                            background: `${phase.color}14`,
                            border: `1px solid ${phase.color}28`,
                            borderRadius: 4,
                            color: phase.accent,
                          }}
                        >
                          {c}
                        </span>
                      ))}
                    </div>
                  </div>

                  {/* Readings */}
                  <div style={{ marginBottom: 14 }}>
                    <div
                      style={{
                        fontSize: 9,
                        letterSpacing: 1.5,
                        color: "#52525B",
                        marginBottom: 8,
                      }}
                    >
                      {t("phase.references")}
                    </div>
                    {phase.readings.map((r, i) => (
                      <div
                        key={i}
                        style={{
                          fontSize: 11,
                          color: "#71717A",
                          marginBottom: 4,
                          paddingLeft: 10,
                          borderLeft: "1px solid #27272A",
                        }}
                      >
                        {r}
                      </div>
                    ))}
                  </div>

                  {/* Deliverable */}
                  <div style={{ marginBottom: 14 }}>
                    <div
                      style={{
                        fontSize: 9,
                        letterSpacing: 1.5,
                        color: "#52525B",
                        marginBottom: 8,
                      }}
                    >
                      {t("phase.deliverable")}
                    </div>
                    <div
                      style={{
                        fontSize: 12,
                        fontWeight: 600,
                        color: "#E4E4E7",
                        marginBottom: 4,
                      }}
                    >
                      {phase.deliverable.name}
                    </div>
                    <div
                      style={{
                        fontSize: 11,
                        color: "#71717A",
                        marginBottom: 10,
                      }}
                    >
                      {phase.deliverable.desc}
                    </div>
                    <div
                      style={{
                        fontSize: 9,
                        letterSpacing: 1.5,
                        color: "#52525B",
                        marginBottom: 6,
                      }}
                    >
                      {t("phase.acceptance")}
                    </div>
                    {phase.deliverable.acceptance.map((a, i) => (
                      <div
                        key={i}
                        style={{
                          fontSize: 11,
                          color: "#A1A1AA",
                          marginBottom: 3,
                          paddingLeft: 14,
                          position: "relative",
                        }}
                      >
                        <span
                          style={{
                            position: "absolute",
                            left: 0,
                            color: phase.accent,
                          }}
                        >
                          ✓
                        </span>
                        {a}
                      </div>
                    ))}
                  </div>

                  {/* Enter lesson button */}
                  {hasLessons(phase.id, locale) && (
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        onEnterLesson(phase.id);
                      }}
                      style={{
                        marginTop: 8,
                        padding: "8px 20px",
                        fontSize: 12,
                        fontFamily: "inherit",
                        fontWeight: 600,
                        background: `${phase.color}22`,
                        border: `1px solid ${phase.color}44`,
                        borderRadius: 6,
                        color: phase.accent,
                        cursor: "pointer",
                        transition: "all 0.15s",
                      }}
                      onMouseEnter={(e) => {
                        e.currentTarget.style.background = `${phase.color}33`;
                      }}
                      onMouseLeave={(e) => {
                        e.currentTarget.style.background = `${phase.color}22`;
                      }}
                    >
                      {t("phase.enter")}
                    </button>
                  )}
                </div>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
};

/* ─── Architecture Tab ─── */

const ArchitectureTab: React.FC<{ t: (key: string) => string }> = ({ t }) => {
  const layers = architecture.layers;

  return (
    <div style={{ animation: "fadeIn 0.2s ease" }}>
      <p style={{ fontSize: 12, color: "#71717A", marginBottom: 24 }}>
        {t("arch.desc")}
      </p>

      {/* Layers (bottom to top → render reversed) */}
      <div style={{ display: "flex", flexDirection: "column-reverse", gap: 6 }}>
        {layers.map((layer: ArchitectureLayer, idx: number) => (
          <div
            key={idx}
            style={{
              display: "flex",
              alignItems: "flex-start",
              gap: 12,
            }}
          >
            {/* Layer label */}
            <div
              style={{
                width: 180,
                flexShrink: 0,
                padding: "8px 12px",
                background: `${layer.color}18`,
                border: `1px solid ${layer.color}33`,
                borderRadius: 6,
                fontSize: 11,
                fontWeight: 600,
                color: layer.color,
                textAlign: "right",
              }}
            >
              {layer.name}
            </div>
            {/* Module chips */}
            <div
              style={{
                display: "flex",
                flexWrap: "wrap",
                gap: 6,
                flex: 1,
                padding: "4px 0",
              }}
            >
              {layer.modules.map((mod: string, mi: number) => (
                <span
                  key={mi}
                  style={{
                    fontSize: 10,
                    padding: "4px 10px",
                    background: `${layer.color}0A`,
                    border: `1px solid ${layer.color}18`,
                    borderRadius: 4,
                    color: `${layer.color}CC`,
                  }}
                >
                  {mod}
                </span>
              ))}
            </div>
          </div>
        ))}
      </div>

      {/* Data flow */}
      <div style={{ marginTop: 32 }}>
        <div
          style={{
            fontSize: 9,
            letterSpacing: 1.5,
            color: "#52525B",
            marginBottom: 12,
          }}
        >
          {t("arch.dataflow")}
        </div>
        <div
          style={{
            display: "flex",
            flexWrap: "wrap",
            alignItems: "center",
            gap: 4,
          }}
        >
          {dataFlowSteps.map((step: string, i: number) => (
            <React.Fragment key={i}>
              <span
                style={{
                  fontSize: 10,
                  padding: "3px 10px",
                  background: "#18181B",
                  border: "1px solid #27272A",
                  borderRadius: 4,
                  color: "#A1A1AA",
                }}
              >
                {step}
              </span>
              {i < dataFlowSteps.length - 1 && (
                <span style={{ color: "#3F3F46", fontSize: 10 }}>→</span>
              )}
            </React.Fragment>
          ))}
        </div>
      </div>
    </div>
  );
};

/* ─── Principles Tab ─── */

const PrinciplesTab: React.FC<{
  principles: Principle[];
  t: (key: string) => string;
}> = ({ principles, t }) => {
  return (
    <div style={{ animation: "fadeIn 0.2s ease" }}>
      <p style={{ fontSize: 12, color: "#71717A", marginBottom: 24 }}>
        {t("principles.desc")}
      </p>

      <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
        {principles.map((p: Principle) => (
          <div
            key={p.num}
            style={{
              display: "flex",
              gap: 16,
              alignItems: "flex-start",
              padding: "16px 20px",
              background: "#111113",
              border: "1px solid #1E1E22",
              borderRadius: 8,
            }}
          >
            <div
              style={{
                fontSize: 32,
                fontWeight: 800,
                color: `${p.color}33`,
                lineHeight: 1,
                flexShrink: 0,
                width: 48,
              }}
            >
              {p.num}
            </div>
            <div>
              <div
                style={{
                  fontSize: 14,
                  fontWeight: 600,
                  color: "#E4E4E7",
                  marginBottom: 6,
                }}
              >
                {p.title}
              </div>
              <div style={{ fontSize: 12, color: "#71717A", lineHeight: 1.6 }}>
                {p.desc}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default CourseRoadmap;
