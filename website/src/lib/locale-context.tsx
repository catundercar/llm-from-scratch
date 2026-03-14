"use client";

import { createContext, useContext } from "react";

type LocaleContextType = {
  locale: string;
};

const LocaleContext = createContext<LocaleContextType>({ locale: "en" });

export function LocaleProvider({
  locale,
  children,
}: {
  locale: string;
  children: React.ReactNode;
}) {
  return (
    <LocaleContext.Provider value={{ locale }}>
      {children}
    </LocaleContext.Provider>
  );
}

export function useLocale() {
  return useContext(LocaleContext);
}
