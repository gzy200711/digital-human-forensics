import { ArrowDownRight, ArrowUpRight, Minus } from "lucide-react";

import type { EvidenceItem } from "@/types/report";

export function EvidenceCard({ item }: { item: EvidenceItem }) {
  const tone =
    item.direction === "指向生成"
      ? "badge-suspect"
      : item.direction === "指向真实"
        ? "badge-trust"
        : "badge-neutral";
  const DirectionIcon =
    item.direction === "指向生成"
      ? ArrowUpRight
      : item.direction === "指向真实"
        ? ArrowDownRight
        : Minus;
  return (
    <article className="evidence-card" data-available={item.available}>
      <div className="evidence-top">
        <div style={{ minWidth: 0 }}>
          <div className="evidence-name">
            {item.display_name}
            {item.layer ? (
              <span className="badge badge-neutral evidence-layer">{item.layer}</span>
            ) : null}
          </div>
          <div className="evidence-id">
            证据 ID <span className="inline-code">{item.id}</span>
            {item.source ? <> · 来源 {item.source}</> : null}
          </div>
        </div>
        <span className={`badge ${tone}`}>
          <DirectionIcon size={12} />
          {item.available ? `${item.direction} · ${item.strength}` : "不可用"}
        </span>
      </div>
      <div className="evidence-meta">
        <span>
          分数{" "}
          {item.score === null
            ? "不可用"
            : `${item.score.toFixed(4)}${item.unit ? ` ${formatUnit(item.unit)}` : ""}`}
        </span>
        <span>
          时间范围{" "}
          {item.time_range
            ? `${formatSeconds(item.time_range.start)}–${formatSeconds(item.time_range.end)}`
            : "仅视频级"}
        </span>
        <span>区域 {item.region?.label ?? "整段媒体"}</span>
        <span>
          模型 {item.model ? `${item.model.name} @ ${item.model.version}` : "未登记"}
        </span>
      </div>
      <div className="evidence-explanation">{item.explanation}</div>
      {Object.keys(item.raw_value).length > 0 ? (
        <div className="evidence-values">
          {Object.entries(item.raw_value)
            .filter(
              ([key, value]) =>
                value !== null &&
                value !== undefined &&
                !["model", "latency_ms"].includes(key),
            )
            .slice(0, 6)
            .map(([key, value]) => (
              <span key={key}>
                {key}: {formatValue(value)}
              </span>
            ))}
        </div>
      ) : null}
      {item.uncertainty || (item.limitations ?? []).length > 0 ? (
        <div className="evidence-limits">
          {item.uncertainty ? <div>不确定度：{item.uncertainty}</div> : null}
          {(item.limitations ?? []).map((limitation) => (
            <div key={limitation}>限制：{limitation}</div>
          ))}
        </div>
      ) : null}
    </article>
  );
}

function formatSeconds(value: number): string {
  const totalTenths = Math.max(0, Math.round(value * 10));
  const minutes = Math.floor(totalTenths / 600);
  const seconds = (totalTenths % 600) / 10;
  return `${String(minutes).padStart(2, "0")}:${seconds.toFixed(1).padStart(4, "0")}`;
}

function formatUnit(unit: string): string {
  if (unit === "probability") return "概率分";
  if (unit === "score") return "分数";
  return unit;
}

function formatValue(value: unknown): string {
  if (typeof value === "number") {
    return Number.isInteger(value) ? String(value) : value.toFixed(4);
  }
  if (typeof value === "boolean") return value ? "是" : "否";
  if (typeof value === "string") return value;
  return JSON.stringify(value).slice(0, 100);
}
