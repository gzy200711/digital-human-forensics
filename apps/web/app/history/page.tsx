import type { Metadata } from "next";

import { AppHeader } from "@/components/app-header";
import { HistoryView } from "@/components/history-view";

export const metadata: Metadata = {
  title: "历史检测 | 数字人鉴别平台",
  description: "查看数字人鉴别平台的历史检测任务、综合判定与报告。",
};

export default function HistoryPage() {
  return (
    <div className="app-shell">
      <AppHeader />
      <main className="page">
        <HistoryView />
      </main>
    </div>
  );
}
