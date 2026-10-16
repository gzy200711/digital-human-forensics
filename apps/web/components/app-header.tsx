import { ScanSearch } from "lucide-react";
import Link from "next/link";

export function AppHeader() {
  return (
    <header className="topbar">
      <Link className="brand" href="/">
        <span className="brand-mark" aria-hidden="true">
          <ScanSearch size={17} />
        </span>
        <span>数字人鉴别平台</span>
      </Link>
      <div className="topbar-meta">
        <span className="live-dot" aria-hidden="true" />
        <span>技术检测服务</span>
        <span>v0.1</span>
      </div>
    </header>
  );
}

