"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";

export default function RootPage() {
  const router = useRouter();
  useEffect(() => {
    router.replace("/en");
  }, [router]);

  return (
    <div className="flex min-h-screen items-center justify-center bg-[var(--bg-primary)]">
      <div className="h-8 w-8 border-2 border-[var(--text-tertiary)] border-t-transparent rounded-full animate-spin" />
    </div>
  );
}
