"use client";

import { useEffect, useState } from "react";
import { Sidebar } from "@/components/shell/sidebar";

const sidebarWidthKey = "rag-console.sidebar-width";
const defaultSidebarWidth = 288;
const minSidebarWidth = 232;
const maxSidebarWidth = 440;

export function AppShell({ children }: { children: React.ReactNode }) {
  const [sidebarWidth, setSidebarWidth] = useState(defaultSidebarWidth);

  useEffect(() => {
    const stored = Number(localStorage.getItem(sidebarWidthKey));
    if (Number.isFinite(stored)) {
      setSidebarWidth(clampSidebarWidth(stored));
    }
  }, []);

  function startSidebarResize(event: React.PointerEvent) {
    event.preventDefault();
    const startX = event.clientX;
    const startWidth = sidebarWidth;

    function handlePointerMove(moveEvent: PointerEvent) {
      const nextWidth = clampSidebarWidth(startWidth + moveEvent.clientX - startX);
      setSidebarWidth(nextWidth);
      localStorage.setItem(sidebarWidthKey, String(nextWidth));
    }

    function handlePointerUp() {
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
      window.removeEventListener("pointermove", handlePointerMove);
      window.removeEventListener("pointerup", handlePointerUp);
    }

    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
    window.addEventListener("pointermove", handlePointerMove);
    window.addEventListener("pointerup", handlePointerUp);
  }

  function updateSidebarWidth(width: number) {
    const nextWidth = clampSidebarWidth(width);
    setSidebarWidth(nextWidth);
    localStorage.setItem(sidebarWidthKey, String(nextWidth));
  }

  return (
    <div
      className="h-screen overflow-hidden bg-background"
      style={{ "--sidebar-width": `${sidebarWidth}px` } as React.CSSProperties}
    >
      <Sidebar
        width={sidebarWidth}
        onResizeStart={startSidebarResize}
        onWidthChange={updateSidebarWidth}
      />
      <main className="no-scrollbar h-screen overflow-y-auto px-4 pb-8 pt-16 md:ml-[var(--sidebar-width)] md:px-6 md:pt-6">
        <div className="mx-auto h-full w-full max-w-[1500px]">{children}</div>
      </main>
    </div>
  );
}

function clampSidebarWidth(width: number) {
  return Math.min(maxSidebarWidth, Math.max(minSidebarWidth, Math.round(width)));
}
