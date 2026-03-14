import React, { createContext, useContext, useState, useCallback, useEffect } from "react";
import zhTW from "./zh-TW";
import zhCN from "./zh-CN";
import en from "./en";

export type Locale = "zh-TW" | "zh-CN" | "en";

const bundles: Record<Locale, Record<string, string>> = {
  "zh-TW": zhTW,
  "zh-CN": zhCN,
  en,
};

interface LocaleCtx {
  locale: Locale;
  setLocale: (l: Locale) => void;
  t: (key: string, vars?: Record<string, string | number>) => string;
}

const LocaleContext = createContext<LocaleCtx>({
  locale: "zh-CN",
  setLocale: () => {},
  t: (k) => k,
});

export const useLocale = () => useContext(LocaleContext);

const STORAGE_KEY = "llm-course-locale";

export const LocaleProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [locale, setLocaleState] = useState<Locale>(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved && (saved === "zh-TW" || saved === "zh-CN" || saved === "en")) return saved as Locale;
    } catch {}
    return "zh-CN";
  });

  const setLocale = useCallback((l: Locale) => {
    setLocaleState(l);
    try { localStorage.setItem(STORAGE_KEY, l); } catch {}
  }, []);

  useEffect(() => {
    try { localStorage.setItem(STORAGE_KEY, locale); } catch {}
  }, [locale]);

  const t = useCallback((key: string, vars?: Record<string, string | number>) => {
    let str = bundles[locale]?.[key] ?? bundles["en"]?.[key] ?? key;
    if (vars) {
      Object.entries(vars).forEach(([k, v]) => {
        str = str.replace(`{${k}}`, String(v));
      });
    }
    return str;
  }, [locale]);

  return (
    <LocaleContext.Provider value={{ locale, setLocale, t }}>
      {children}
    </LocaleContext.Provider>
  );
};
