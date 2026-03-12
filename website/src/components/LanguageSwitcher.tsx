import React from "react";
import { useLocale, type Locale } from "../i18n";

const langs: { code: Locale; label: string }[] = [
  { code: "zh-TW", label: "繁" },
  { code: "zh-CN", label: "简" },
  { code: "en", label: "EN" },
];

const LanguageSwitcher: React.FC = () => {
  const { locale, setLocale } = useLocale();

  return (
    <div style={{ display: "flex", gap: 4 }}>
      {langs.map((l) => (
        <button
          key={l.code}
          onClick={() => setLocale(l.code)}
          style={{
            padding: "4px 10px",
            fontSize: 11,
            fontFamily: "inherit",
            border: locale === l.code ? "1px solid #52525B" : "1px solid #27272A",
            borderRadius: 4,
            background: locale === l.code ? "#27272A" : "transparent",
            color: locale === l.code ? "#E4E4E7" : "#71717A",
            cursor: "pointer",
            transition: "all 0.15s",
          }}
        >
          {l.label}
        </button>
      ))}
    </div>
  );
};

export default LanguageSwitcher;
