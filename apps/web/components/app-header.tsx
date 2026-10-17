"use client";

import { History, ScanLine, ScanSearch } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";

export function AppHeader() {
  const pathname = usePathname();
  const historyActive = pathname.startsWith("/history");
  const detectionActive = !historyActive;

  return (
    <header className="topbar">
      <div className="topbar-start">
        <Link className="brand" href="/">
          <span className="brand-mark" aria-hidden="true">
            <ScanSearch size={17} />
          </span>
          <span className="brand-name">数字人鉴别平台</span>
        </Link>
        <nav className="topbar-nav" aria-label="主导航">
          <Link
            href="/"
            data-active={detectionActive}
            aria-current={detectionActive ? "page" : undefined}
          >
            <ScanLine size={14} />
            <span>检测</span>
          </Link>
          <Link
            href="/history"
            data-active={historyActive}
            aria-current={historyActive ? "page" : undefined}
          >
            <History size={14} />
            <span>历史检测</span>
          </Link>
        </nav>
      </div>
      <div className="topbar-meta">
        <span className="live-dot" aria-hidden="true" />
        <span>技术检测服务</span>
        <span>v0.1</span>
      </div>
    </header>
  );
}

