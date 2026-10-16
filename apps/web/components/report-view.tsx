"use client";

import {
  AlertTriangle,
  BadgeCheck,
  CheckCircle2,
  ChevronDown,
  CircleGauge,
  Clock3,
  Download,
  FileJson2,
  Info,
  Layers,
  LoaderCircle,
  MessagesSquare,
  Scale,
  ShieldQuestion,
  XCircle,
} from "lucide-react";
import { useEffect, useState } from "react";

import { ApiError, formatBytes, formatDuration, getJob, getReport, verdictTone } from "@/lib/api";
import type { ForensicReport, JobStatus } from "@/types/report";
import { AppealDialog } from "@/components/appeal-dialog";
import { EvidenceCard } from "@/components/evidence-card";
import {
  AbstentionPanel,
  CoveragePanel,
  ModalityGrid,
} from "@/components/report-overview";
import { VideoEvidence } from "@/components/video-evidence";

export function ReportView({ jobId }: { jobId: string }) {
  const [job, setJob] = useState<JobStatus | null>(null);
  const [report, setReport] = useState<ForensicReport | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;

    async function poll() {
      try {
        const status = await getJob(jobId);
        if (cancelled) return;
        setJob(status);
        setError(null);
        if (status.status === "completed" && status.report_ready) {
          const payload = await getReport(jobId);
          if (!cancelled) setReport(payload);
          return;
        }
        if (status.status === "failed") return;
        timer = setTimeout(poll, 1200);
      } catch (caught) {
        if (cancelled) return;
        setError(
          caught instanceof ApiError
            ? `${caught.code}: ${caught.message}`
            : "无法读取任务状态。",
        );
        timer = setTimeout(poll, 2500);
      }
    }

    void poll();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [jobId]);

  if (error && !job) {
    return (
      <div className="panel">
        <div className="empty-state">
          <div>
            <span className="empty-icon">
              <XCircle size={24} />
            </span>
            <div style={{ fontWeight: 700 }}>任务读取失败</div>
            <div className="muted" style={{ marginTop: 7 }}>
              {error}
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (!job) return <ReportSkeleton />;

  if (job.status === "failed") {
    return (
      <div className="panel">
        <div className="empty-state">
          <div>
            <span className="empty-icon" style={{ background: "#fef3f2", color: "#b42318" }}>
              <AlertTriangle size={24} />
            </span>
            <div style={{ fontWeight: 700 }}>检测失败</div>
            <div className="muted" style={{ marginTop: 7 }}>
              {job.error_code}：{job.error_message}
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (!report) {
    return (
      <>
        <StatusStrip job={job} />
        <ReportSkeleton />
      </>
    );
  }

  const tone = verdictTone(report.verdict);
  const VerdictIcon =
    report.verdict === "可信"
      ? CheckCircle2
      : report.verdict === "疑似生成合成"
        ? AlertTriangle
        : ShieldQuestion;
  return (
    <>
      <div className="page-header">
        <div>
          <div className="page-eyebrow">Forensic Report</div>
          <h1 className="page-title">{report.media.filename}</h1>
          <p className="page-subtitle">
            报告 {report.report_id} · 任务 {report.job_id}
          </p>
        </div>
        <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
          {report.artifacts.json_report_url ? (
            <a
              className="button button-secondary"
              href={report.artifacts.json_report_url}
              target="_blank"
              rel="noreferrer"
            >
              <FileJson2 size={15} />
              JSON
            </a>
          ) : null}
          {report.artifacts.html_report_url ? (
            <a
              className="button button-secondary"
              href={report.artifacts.html_report_url}
              target="_blank"
              rel="noreferrer"
            >
              <Download size={15} />
              静态报告
            </a>
          ) : null}
          <AppealDialog jobId={jobId} />
        </div>
      </div>

      <section className="verdict-band" data-verdict={tone}>
        <div>
          <div className="verdict-label">综合判定</div>
          <div className="verdict-value">
            <VerdictIcon
              size={27}
              style={{
                verticalAlign: "-3px",
                marginRight: 9,
                color:
                  tone === "trusted"
                    ? "var(--trust)"
                    : tone === "suspected"
                      ? "var(--suspect)"
                      : "var(--uncertain)",
              }}
            />
            {report.verdict}
          </div>
          <div className="verdict-summary">{report.summary}</div>
          <div className="verdict-scope">
            <Info size={14} />
            <span>{scopeStatement(report.verdict)}</span>
          </div>
        </div>
        <div className="score-block">
          <div className="score-head">
            <span className="score-number">
              {(report.calibrated_probability?.value ?? report.calibrated_score).toFixed(3)}
            </span>
            <span
              className={`badge ${
                report.calibrated_probability?.is_calibrated ? "badge-brand" : "badge-neutral"
              }`}
            >
              {report.calibrated_probability?.is_calibrated ? "已校准" : "未校准"}
            </span>
          </div>
          <div className="score-label">
            {report.calibrated_probability?.is_calibrated
              ? "校准生成风险概率"
              : "bootstrap 风险分（未校准）"}
          </div>
          <div className="muted score-meta">
            置信区间{" "}
            {report.calibrated_probability?.confidence_interval
              ? `[${report.calibrated_probability.confidence_interval[0].toFixed(3)}, ${report.calibrated_probability.confidence_interval[1].toFixed(3)}]`
              : "未提供（样本量不足）"}
          </div>
          <div className="muted score-meta">
            判定强度 {(report.confidence * 100).toFixed(1)}%（不是置信概率）
          </div>
          {report.gray_zone ? (
            <div className="gray-zone">
              <div className="gray-zone-track" aria-hidden="true">
                <span
                  className="gray-zone-band"
                  style={{
                    left: `${report.gray_zone.minimum * 100}%`,
                    width: `${Math.max(
                      2,
                      (report.gray_zone.maximum - report.gray_zone.minimum) * 100,
                    )}%`,
                  }}
                />
                <span
                  className="gray-zone-marker"
                  style={{
                    left: `${Math.min(
                      100,
                      Math.max(
                        0,
                        (report.gray_zone.score ?? report.calibrated_score) * 100,
                      ),
                    )}%`,
                  }}
                />
              </div>
              <div className="gray-zone-caption">
                灰区 {report.gray_zone.minimum.toFixed(2)}–
                {report.gray_zone.maximum.toFixed(2)}
                {report.gray_zone.in_gray_zone ? " · 当前落在灰区" : ""}
              </div>
            </div>
          ) : null}
          <div className="muted score-meta">
            校准分布：{report.calibrated_probability?.distribution ?? "未声明"}
          </div>
        </div>
      </section>

      <ModalityGrid report={report} />

      {report.verdict === "无法确定" ? (
        <AbstentionPanel reasons={report.abstention_reasons} />
      ) : null}

      <div className="report-grid">
        <div className="stack">
          <VideoEvidence report={report} />

          <section className="panel">
            <div className="panel-header">
              <h2 className="panel-title">证据链</h2>
              <span className="badge badge-neutral">{report.evidence.length} 项</span>
            </div>
            <div className="panel-body">
              <div className="evidence-list">
                {report.evidence.map((item) => (
                  <EvidenceCard item={item} key={item.id} />
                ))}
              </div>
            </div>
          </section>

          {report.review?.triggered ? (
            <section className="panel">
              <div className="panel-header">
                <h2 className="panel-title">证据复核</h2>
                <span
                  className={`badge ${
                    report.review.status === "completed" ? "badge-brand" : "badge-uncertain"
                  }`}
                >
                  <MessagesSquare size={13} />
                  {report.review.provider}
                </span>
              </div>
              <div className="panel-body">
                {report.review.status === "unavailable" ? (
                  <div className="alert alert-error" style={{ marginBottom: 14 }}>
                    {report.review.summary}
                  </div>
                ) : (
                  <div className="review-list">
                    {report.review.statements.map((statement, index) => (
                      <div className="review-item" key={`${statement.kind}-${index}`}>
                        <div className="review-item-top">
                          <span className={`badge ${reviewKindTone(statement.kind)}`}>
                            {statement.kind}
                          </span>
                          <span className="review-citation">
                            证据 {statement.evidence_ids.join(" / ")}
                          </span>
                        </div>
                        <div className="review-claim">{statement.claim}</div>
                      </div>
                    ))}
                  </div>
                )}
                {report.review.rejected_claim_count > 0 ? (
                  <div className="muted" style={{ marginTop: 12, fontSize: 12 }}>
                    已拒绝 {report.review.rejected_claim_count} 条无法回溯到现有证据的陈述。
                  </div>
                ) : null}
                <div className="review-note">{report.review.note}</div>
              </div>
            </section>
          ) : null}

          <section className="panel">
            <div className="panel-header">
              <h2 className="panel-title">时间定位</h2>
              <Clock3 size={17} color="#667085" />
            </div>
            <div className="panel-body">
              {report.time_ranges.length > 0 ? (
                <div className="stack" style={{ gap: 10 }}>
                  {report.time_ranges.map((range, index) => (
                    <div
                      className="ops-row"
                      key={`${range.start}-${range.end}-${index}`}
                    >
                      <div>
                        <div style={{ fontWeight: 700 }}>
                          {formatSeconds(range.start)}–{formatSeconds(range.end)}
                        </div>
                        <div className="muted" style={{ marginTop: 4 }}>
                          {range.reason}
                        </div>
                      </div>
                      <span className="badge badge-suspect">{range.score.toFixed(3)}</span>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="muted">未形成足够稳定的疑似时间段。</div>
              )}
            </div>
          </section>
        </div>

        <aside className="stack">
          <section className="panel">
            <div className="panel-header">
              <h2 className="panel-title">媒体摘要</h2>
              <CircleGauge size={17} color="#667085" />
            </div>
            <div className="panel-body">
              <div className="metric-grid">
                <Metric label="时长" value={formatDuration(report.media.duration_seconds)} />
                <Metric
                  label="分辨率"
                  value={
                    report.media.width && report.media.height
                      ? `${report.media.width}×${report.media.height}`
                      : "未知"
                  }
                />
                <Metric label="文件大小" value={formatBytes(report.media.size_bytes)} />
              </div>
              <table className="data-table" style={{ marginTop: 16 }}>
                <tbody>
                  <tr>
                    <th>视频编码</th>
                    <td>{report.media.video_codec ?? "无"}</td>
                  </tr>
                  <tr>
                    <th>音频编码</th>
                    <td>{report.media.audio_codec ?? "无"}</td>
                  </tr>
                  <tr>
                    <th>SHA-256</th>
                    <td className="inline-code">{report.media.sha256}</td>
                  </tr>
                  <tr>
                    <th>建议动作</th>
                    <td>{report.recommended_action}</td>
                  </tr>
                  <tr>
                    <th>审计追踪</th>
                    <td className="inline-code">
                      {report.audit_trace_id ?? report.job_id}
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </section>

          <section className="panel">
            <div className="panel-header">
              <h2 className="panel-title">可鉴定性</h2>
              <span
                className={`badge ${
                  report.assessability.status === "可鉴定"
                    ? "badge-trust"
                    : report.assessability.status === "不可鉴定"
                      ? "badge-suspect"
                      : "badge-uncertain"
                }`}
              >
                {report.assessability.status} · {report.assessability.score.toFixed(3)}
              </span>
            </div>
            <div className="panel-body" style={{ paddingTop: 8, paddingBottom: 10 }}>
              <Accordion title="质量门控因素">
                {report.assessability.reasons.length > 0 ? (
                  <ul className="plain-list">
                    {report.assessability.reasons.map((reason) => (
                      <li key={reason}>{reason}</li>
                    ))}
                  </ul>
                ) : (
                  <div>质量门控未发现需要降级的因素。</div>
                )}
                <div className="stack" style={{ marginTop: 14, gap: 9 }}>
                  {report.assessability.factors.map((factor) => (
                    <div key={factor.name}>
                      <div style={{ display: "flex", justifyContent: "space-between", gap: 12 }}>
                        <strong>{factor.name}</strong>
                        <span className={`badge badge-${factorTone(factor.status)}`}>
                          {String(factor.value)}
                        </span>
                      </div>
                      <div className="muted" style={{ marginTop: 4 }}>
                        {factor.explanation}
                      </div>
                    </div>
                  ))}
                </div>
              </Accordion>
            </div>
          </section>

          <CoveragePanel coverage={report.coverage} />

          <section className="panel">
            <div className="panel-header">
              <h2 className="panel-title">溯源信息</h2>
              <span
                className={`badge ${
                  report.provenance.status === "阳性"
                    ? "badge-suspect"
                    : report.provenance.status === "未检查"
                      ? "badge-uncertain"
                      : "badge-neutral"
                }`}
              >
                {report.provenance.status ?? "无声明"}
              </span>
            </div>
            <div className="panel-body" style={{ paddingTop: 8, paddingBottom: 10 }}>
              {report.provenance.checks.map((check) => (
                <Accordion
                  key={check.name}
                  title={`${check.name} · ${check.status}`}
                  defaultOpen={check.positive}
                >
                  {check.explanation}
                </Accordion>
              ))}
              {report.provenance.limitation ? (
                <div className="provenance-limit">{report.provenance.limitation}</div>
              ) : null}
              <div className="muted" style={{ marginTop: 12, lineHeight: 1.6 }}>
                {report.provenance.note}
              </div>
            </div>
          </section>

          {report.threshold_policy ? (
            <section className="panel">
              <div className="panel-header">
                <h2 className="panel-title">阈值与校准</h2>
                <Scale size={17} color="#667085" />
              </div>
              <div className="panel-body">
                <div className="inline-code" style={{ fontSize: 12 }}>
                  {report.threshold_policy.policy_id}
                </div>
                <div className="muted" style={{ marginTop: 6, fontSize: 12 }}>
                  来源：{report.threshold_policy.source}
                </div>
                <div className="kv-row">
                  <span>灰区</span>
                  <span>
                    {report.threshold_policy.gray_zone[0].toFixed(2)}–
                    {report.threshold_policy.gray_zone[1].toFixed(2)}
                  </span>
                </div>
                <div className="kv-row">
                  <span>目标 FPR</span>
                  <span>
                    {report.threshold_policy.target_fprs
                      .map((value) => `${(value * 100).toFixed(value < 0.01 ? 1 : 0)}%`)
                      .join(" / ")}
                  </span>
                </div>
                <div className="kv-row">
                  <span>真实阴性样本</span>
                  <span>
                    {report.threshold_policy.real_negative_samples === null
                      ? "无记录"
                      : report.threshold_policy.real_negative_samples}
                  </span>
                </div>
                <div
                  className={`alert ${
                    report.threshold_policy.fpr_evidence_status === "verified"
                      ? ""
                      : "alert-error"
                  }`}
                  style={{ marginTop: 12 }}
                >
                  {report.threshold_policy.statement}
                </div>
              </div>
            </section>
          ) : null}

          {report.model_snapshot || report.model_versions.length > 0 ? (
            <section className="panel">
              <div className="panel-header">
                <h2 className="panel-title">模型快照</h2>
                <Layers size={17} color="#667085" />
              </div>
              <div className="panel-body" style={{ paddingTop: 12 }}>
                {report.model_snapshot ? (
                  <div className="model-snapshot">
                    <div className="model-snapshot-top">
                      <span className="inline-code" style={{ fontSize: 12 }}>
                        {report.model_snapshot.registry_version}
                      </span>
                      {report.model_snapshot.is_research_only ? (
                        <span className="badge badge-uncertain">
                          <BadgeCheck size={12} />
                          研究版本
                        </span>
                      ) : null}
                    </div>
                    <div className="muted" style={{ marginTop: 8, lineHeight: 1.6 }}>
                      {report.model_snapshot.note}
                    </div>
                  </div>
                ) : null}
                <div className="stack" style={{ marginTop: 12, gap: 8 }}>
                  {report.model_versions.map((model) => (
                    <div className="model-row" key={`${model.name}-${model.version}`}>
                      <span className="model-name">{model.name}</span>
                      <span className="muted">{model.version}</span>
                      <span className="muted model-license">
                        {model.license ?? "许可证未核验"}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </section>
          ) : null}

          <section className="panel">
            <div className="panel-header">
              <h2 className="panel-title">局限声明</h2>
              <ShieldQuestion size={17} color="#667085" />
            </div>
            <div className="panel-body">
              <ul className="plain-list">
                {report.limitations.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            </div>
          </section>
        </aside>
      </div>
    </>
  );
}

function StatusStrip({ job }: { job: JobStatus }) {
  return (
    <div className="status-strip">
      <div className="status-main">
        <span className="status-icon">
          <LoaderCircle size={18} className="spin" />
        </span>
        <div>
          <div className="status-title">{job.original_filename}</div>
          <div className="status-stage">
            {job.stage} · {job.progress}%
          </div>
        </div>
      </div>
      <div className="progress-track" aria-label="任务进度">
        <div className="progress-fill" style={{ width: `${job.progress}%` }} />
      </div>
    </div>
  );
}

function ReportSkeleton() {
  return (
    <div className="stack">
      <div className="skeleton" style={{ height: 126 }} />
      <div className="report-grid">
        <div className="skeleton" style={{ height: 470 }} />
        <div className="skeleton" style={{ height: 390 }} />
      </div>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric">
      <div className="metric-label">{label}</div>
      <div className="metric-value" style={{ fontSize: 16 }}>
        {value}
      </div>
    </div>
  );
}

function Accordion({
  title,
  children,
  defaultOpen = false,
}: {
  title: string;
  children: React.ReactNode;
  defaultOpen?: boolean;
}) {
  return (
    <details className="accordion" open={defaultOpen}>
      <summary className="accordion-trigger">
        <span>{title}</span>
        <ChevronDown size={16} />
      </summary>
      <div className="accordion-content">{children}</div>
    </details>
  );
}

function factorTone(status: string): string {
  if (status === "good") return "trust";
  if (status === "warning") return "uncertain";
  if (status === "bad") return "suspect";
  return "neutral";
}

function reviewKindTone(kind: string): string {
  if (kind === "生成风险") return "badge-suspect";
  if (kind === "真实性提示") return "badge-trust";
  if (kind === "质量限制") return "badge-uncertain";
  return "badge-neutral";
}

function scopeStatement(verdict: ForensicReport["verdict"]): string {
  if (verdict === "可信") {
    return "“可信”只表示在本次检测覆盖的模态、时段与质量范围内未发现足够强的生成合成证据，不构成对原始真实性的证明。";
  }
  if (verdict === "疑似生成合成") {
    return "该判定表示多个独立信号在同一时段或同一模态指向生成合成，且越过已锁定阈值；不代表对人物身份、创作意图或法律责任的判断。";
  }
  return "该判定表示证据不足、信号互相矛盾、样本超出已验证分布，或可鉴定性不足；平台不对内容真伪给出方向性结论。";
}

function formatSeconds(value: number): string {
  const totalTenths = Math.max(0, Math.round(value * 10));
  const minutes = Math.floor(totalTenths / 600);
  const seconds = (totalTenths % 600) / 10;
  return `${String(minutes).padStart(2, "0")}:${seconds.toFixed(1).padStart(4, "0")}`;
}
