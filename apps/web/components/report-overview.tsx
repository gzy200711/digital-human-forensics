"use client";

import {
  AudioLines,
  Fingerprint,
  ScanFace,
  ShieldQuestion,
  Waves,
} from "lucide-react";

import { verdictTone } from "@/lib/api";
import type {
  ForensicReport,
  ModalityCoverage,
  ModalityKey,
  ModalityVerdict,
} from "@/types/report";

const MODALITY_ICON: Record<ModalityKey, typeof ScanFace> = {
  visual: ScanFace,
  audio: AudioLines,
  cross_modal: Waves,
  provenance: Fingerprint,
};

export function ModalityGrid({ report }: { report: ForensicReport }) {
  const verdicts = report.modality_verdicts ?? [];
  if (verdicts.length === 0) return null;
  return (
    <section className="panel">
      <div className="panel-header">
        <h2 className="panel-title">模态级结论</h2>
        <span className="badge badge-neutral">先分模态，再综合</span>
      </div>
      <div className="panel-body">
        <div className="modality-grid">
          {verdicts.map((item) => (
            <ModalityCard key={item.modality} item={item} report={report} />
          ))}
        </div>
        <div className="muted modality-note">
          不同模态指向不同方向是允许的组合，例如“音频疑似合成、画面暂未发现异常”，
          不会被自动判为信号冲突。
        </div>
      </div>
    </section>
  );
}

function ModalityCard({
  item,
  report,
}: {
  item: ModalityVerdict;
  report: ForensicReport;
}) {
  const Icon = MODALITY_ICON[item.modality] ?? ShieldQuestion;
  const tone = item.status === "unavailable" ? "unavailable" : verdictTone(item.verdict);
  const names = item.signal_ids
    .map((id) => report.evidence.find((entry) => entry.id === id)?.display_name ?? id)
    .join("、");
  const provenanceWithoutDeclaration =
    item.modality === "provenance" && item.status === "available" && item.score === null;
  const scoreText =
    item.score !== null
      ? item.score.toFixed(3)
      : provenanceWithoutDeclaration
        ? "不适用"
        : "无可用分数";
  return (
    <article className="modality-card" data-tone={tone}>
      <div className="modality-top">
        <span className="modality-icon" aria-hidden="true">
          <Icon size={16} />
        </span>
        <span className="modality-label">{item.label}</span>
        <span className={`badge ${badgeTone(tone)}`}>
          {item.status === "unavailable"
            ? "分支不可用"
            : provenanceWithoutDeclaration
              ? "无生成声明"
              : item.verdict}
        </span>
      </div>
      <div className="modality-score">
        {scoreText}
        <span className="modality-score-unit">
          {item.score !== null
            ? " 分支分"
            : provenanceWithoutDeclaration
              ? " 溯源分支不产生风险分"
              : ""}
        </span>
      </div>
      <CoverageLine value={item.coverage} label="本模态覆盖" />
      <p className="modality-summary">{item.summary}</p>
      {names ? <div className="modality-signals">信号：{names}</div> : null}
      {item.status !== "available" && item.limitations.length > 0 ? (
        <div className="modality-limit">{item.limitations[0]}</div>
      ) : null}
    </article>
  );
}

export function CoveragePanel({ coverage }: { coverage?: ModalityCoverage }) {
  if (!coverage) return null;
  const rows: Array<[string, number | null, string]> = [
    ["人脸区域", coverage.face, "被采样帧中检出可用人脸的比例"],
    ["音频", coverage.audio, "音轨中被音频分支有效分析的时长占比"],
    ["跨模态", coverage.cross_modal, "唇音同步可对齐的时长占比"],
    ["全画面", coverage.full_frame, "当前版本没有独立的全画面分支"],
  ];
  return (
    <section className="panel">
      <div className="panel-header">
        <h2 className="panel-title">覆盖率</h2>
        <span className="badge badge-neutral">结论只在此范围内成立</span>
      </div>
      <div className="panel-body" style={{ paddingTop: 8, paddingBottom: 12 }}>
        {rows.map(([label, value, hint]) => (
          <div className="coverage-row" key={label} title={hint}>
            <div className="coverage-row-top">
              <span>{label}</span>
              <span className="coverage-value">
                {value === null || value === undefined
                  ? "无法估计"
                  : `${Math.round(value * 100)}%`}
              </span>
            </div>
            <div className="coverage-track" aria-label={`${label}覆盖率`}>
              <div
                className="coverage-fill"
                data-unknown={value === null || value === undefined}
                style={{
                  width:
                    value === null || value === undefined
                      ? "100%"
                      : `${Math.max(2, Math.round(value * 100))}%`,
                }}
              />
            </div>
          </div>
        ))}
        {(coverage.notes ?? []).length > 0 ? (
          <ul className="plain-list coverage-notes">
            {coverage.notes.map((note) => (
              <li key={note}>{note}</li>
            ))}
          </ul>
        ) : null}
      </div>
    </section>
  );
}

export function AbstentionPanel({ reasons }: { reasons?: string[] }) {
  if (!reasons || reasons.length === 0) return null;
  return (
    <section className="panel panel-uncertain">
      <div className="panel-header">
        <h2 className="panel-title">为什么判定为“无法确定”</h2>
        <span className="badge badge-uncertain">
          <ShieldQuestion size={13} />
          拒判原因
        </span>
      </div>
      <div className="panel-body">
        <ul className="plain-list abstention-list">
          {reasons.map((reason) => (
            <li key={reason}>{reason}</li>
          ))}
        </ul>
      </div>
    </section>
  );
}

function CoverageLine({ value, label }: { value: number | null; label: string }) {
  return (
    <div className="coverage-row coverage-row-compact">
      <div className="coverage-row-top">
        <span>{label}</span>
        <span className="coverage-value">
          {value === null ? "无法估计" : `${Math.round(value * 100)}%`}
        </span>
      </div>
      <div className="coverage-track">
        <div
          className="coverage-fill"
          data-unknown={value === null}
          style={{
            width: value === null ? "100%" : `${Math.max(2, Math.round(value * 100))}%`,
          }}
        />
      </div>
    </div>
  );
}

function badgeTone(tone: string): string {
  if (tone === "trusted") return "badge-trust";
  if (tone === "suspected") return "badge-suspect";
  if (tone === "unavailable") return "badge-neutral";
  return "badge-uncertain";
}
