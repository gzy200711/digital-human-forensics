"use client";

import {
  AlertTriangle,
  ArrowLeft,
  CheckCircle2,
  Download,
  FileQuestion,
  LoaderCircle,
  XCircle,
} from "lucide-react";
import Link from "next/link";
import { useEffect, useMemo, useRef, useState } from "react";

import { MAX_BATCH_JOBS, formatBytes, getJob, getReport } from "@/lib/api";
import type { ForensicReport, JobStatus } from "@/types/report";

const POLL_INTERVAL_MS = 2000;
/** 状态连续读取失败约 20 秒后判定该任务失联。 */
const STATUS_RETRY_LIMIT = 10;
/** 报告读取失败重试次数，超过后在汇总表标记为读取失败。 */
const REPORT_RETRY_LIMIT = 3;

interface BatchViewProps {
  idsParam: string;
  failedParam: string;
}

function verdictBadgeClass(verdict: ForensicReport["verdict"]): string {
  if (verdict === "可信") return "badge-trust";
  if (verdict === "疑似生成合成") return "badge-suspect";
  return "badge-uncertain";
}

function statusLabel(status: JobStatus["status"] | undefined): string {
  switch (status) {
    case "queued":
      return "排队中";
    case "processing":
      return "检测中";
    case "completed":
      return "已完成";
    case "failed":
      return "失败";
    default:
      return "读取中";
  }
}

function decodeFailedNames(value: string): string[] {
  return value
    .split("|")
    .map((part) => {
      try {
        return decodeURIComponent(part);
      } catch {
        return part;
      }
    })
    .filter(Boolean);
}

function escapeCsv(value: string | number): string {
  return `"${String(value ?? "").replace(/"/g, '""')}"`;
}

export function BatchView({ idsParam, failedParam }: BatchViewProps) {
  const jobIds = useMemo(
    () =>
      idsParam
        .split(",")
        .map((value) => value.trim())
        .filter(Boolean)
        .slice(0, MAX_BATCH_JOBS),
    [idsParam],
  );
  const failedSubmissions = useMemo(() => decodeFailedNames(failedParam), [failedParam]);
  const idsKey = jobIds.join(",");

  const [statuses, setStatuses] = useState<Record<string, JobStatus>>({});
  const [reports, setReports] = useState<Record<string, ForensicReport>>({});
  const [lostJobs, setLostJobs] = useState<Record<string, boolean>>({});
  const [lostReports, setLostReports] = useState<Record<string, boolean>>({});
  const reportsRef = useRef<Record<string, ForensicReport>>({});
  const lostJobsRef = useRef<Record<string, boolean>>({});
  const lostReportsRef = useRef<Record<string, boolean>>({});

  useEffect(() => {
    if (jobIds.length === 0) return;
    let cancelled = false;
    let timer: number | undefined;
    const statusFailures = new Map<string, number>();
    const reportFailures = new Map<string, number>();

    function markLost(ref: Record<string, boolean>, id: string, apply: (next: Record<string, boolean>) => void) {
      if (ref[id]) return;
      ref[id] = true;
      apply({ ...ref });
    }

    async function tick() {
      const results = await Promise.all(
        jobIds.map(async (id) => {
          try {
            return await getJob(id);
          } catch {
            return null;
          }
        }),
      );
      if (cancelled) return;

      const nextStatuses: Record<string, JobStatus> = {};
      results.forEach((status, index) => {
        const id = jobIds[index];
        if (status) {
          nextStatuses[id] = status;
          statusFailures.delete(id);
        } else if (!lostJobsRef.current[id]) {
          const count = (statusFailures.get(id) ?? 0) + 1;
          statusFailures.set(id, count);
          if (count >= STATUS_RETRY_LIMIT) {
            markLost(lostJobsRef.current, id, setLostJobs);
          }
        }
      });
      setStatuses((previous) => ({ ...previous, ...nextStatuses }));

      const needReports = results.filter(
        (status): status is JobStatus =>
          Boolean(status) &&
          status!.status === "completed" &&
          status!.report_ready &&
          !reportsRef.current[status!.job_id] &&
          !lostReportsRef.current[status!.job_id],
      );
      if (needReports.length > 0) {
        const fetched = await Promise.all(
          needReports.map(async (status) => {
            try {
              return await getReport(status.job_id);
            } catch {
              return null;
            }
          }),
        );
        if (cancelled) return;
        const additions: Record<string, ForensicReport> = {};
        fetched.forEach((report, index) => {
          const id = needReports[index].job_id;
          if (report) {
            additions[id] = report;
            reportFailures.delete(id);
          } else {
            const count = (reportFailures.get(id) ?? 0) + 1;
            reportFailures.set(id, count);
            if (count >= REPORT_RETRY_LIMIT) {
              markLost(lostReportsRef.current, id, setLostReports);
            }
          }
        });
        if (Object.keys(additions).length > 0) {
          reportsRef.current = { ...reportsRef.current, ...additions };
          setReports(reportsRef.current);
        }
      }

      const jobsSettled = jobIds.every((id, index) => {
        const status = results[index] ?? null;
        return (
          lostJobsRef.current[id] ||
          status?.status === "completed" ||
          status?.status === "failed"
        );
      });
      const reportsSettled = jobIds.every((id, index) => {
        const status = results[index] ?? null;
        if (status?.status !== "completed" || !status.report_ready) return true;
        return Boolean(reportsRef.current[id]) || Boolean(lostReportsRef.current[id]);
      });
      if (!(jobsSettled && reportsSettled)) {
        timer = window.setTimeout(tick, POLL_INTERVAL_MS);
      }
    }

    void tick();
    return () => {
      cancelled = true;
      if (timer) window.clearTimeout(timer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [idsKey]);

  const rows = jobIds.map((id, index) => ({
    id,
    index: index + 1,
    status: statuses[id],
    report: reports[id],
    lost: Boolean(lostJobs[id]),
    reportLost: Boolean(lostReports[id]),
  }));

  if (rows.length === 0) {
    return (
      <div className="panel">
        <div className="empty-state">
          <div>
            <span className="empty-icon">
              <FileQuestion size={24} />
            </span>
            <div style={{ fontWeight: 700 }}>链接中缺少任务列表</div>
            <div className="muted" style={{ marginTop: 7 }}>
              请从提交页选择多个文件或粘贴多条 URL 发起批量检测。
            </div>
            <div style={{ marginTop: 16 }}>
              <Link className="button button-secondary" href="/">
                <ArrowLeft size={15} />
                返回提交页
              </Link>
            </div>
          </div>
        </div>
      </div>
    );
  }

  const completedCount = rows.filter((row) => row.status?.status === "completed").length;
  const failedCount = rows.filter(
    (row) => row.status?.status === "failed" || row.lost,
  ).length;
  const finishedCount = completedCount + failedCount;
  const runningCount = rows.length - finishedCount;
  const allFinished = finishedCount === rows.length;
  const overallProgress = Math.round(
    rows.reduce((sum, row) => {
      if (row.status?.status === "completed" || row.lost) return sum + 100;
      return sum + (row.status?.progress ?? 0);
    }, 0) / rows.length,
  );
  const verdictCounts = { 可信: 0, 疑似生成合成: 0, 无法确定: 0 };
  rows.forEach((row) => {
    if (row.report) verdictCounts[row.report.verdict] += 1;
  });

  function exportCsv() {
    const origin = window.location.origin;
    const header = [
      "序号",
      "文件名",
      "文件大小",
      "状态",
      "综合判定",
      "判定强度(%)",
      "建议动作",
      "错误信息",
      "任务ID",
      "报告链接",
    ];
    const lines = rows.map((row) => [
      row.index,
      row.status?.original_filename ?? "",
      row.status ? formatBytes(row.status.size_bytes) : "",
      row.lost ? "读取失败" : statusLabel(row.status?.status),
      row.report?.verdict ?? (row.reportLost ? "报告读取失败" : ""),
      row.report ? (row.report.confidence * 100).toFixed(1) : "",
      row.report?.recommended_action ?? "",
      row.status?.status === "failed"
        ? `${row.status.error_code ?? ""} ${row.status.error_message ?? ""}`.trim()
        : row.lost
          ? "任务状态读取失败"
          : "",
      row.id,
      `${origin}/jobs/${row.id}`,
    ]);
    const csv = [header, ...lines]
      .map((columns) => columns.map(escapeCsv).join(","))
      .join("\r\n");
    const blob = new Blob(["\uFEFF" + csv], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    const stamp = new Date()
      .toISOString()
      .slice(0, 19)
      .replace(/[T:]/g, "-");
    anchor.href = url;
    anchor.download = `批量鉴别结果-${stamp}.csv`;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    URL.revokeObjectURL(url);
  }

  return (
    <>
      <div className="page-header">
        <div>
          <div className="page-eyebrow">Batch Forensic Screening</div>
          <h1 className="page-title">批量鉴别结果</h1>
          <p className="page-subtitle">
            共 {rows.length} 个任务 · 已完成 {completedCount} · 进行中 {runningCount} · 失败{" "}
            {failedCount}
          </p>
        </div>
        <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
          <Link className="button button-secondary" href="/">
            <ArrowLeft size={15} />
            新的检测
          </Link>
          <button
            className="button button-primary"
            type="button"
            onClick={exportCsv}
            disabled={Object.keys(reports).length === 0}
          >
            <Download size={15} />
            导出 CSV
          </button>
        </div>
      </div>

      {failedSubmissions.length > 0 ? (
        <div className="alert alert-error" role="alert" style={{ marginBottom: 16 }}>
          <AlertTriangle size={15} style={{ verticalAlign: "-2px", marginRight: 6 }} />
          以下 {failedSubmissions.length} 个文件未能成功提交，未纳入本次批量检测：
          {failedSubmissions.join("、")}
        </div>
      ) : null}

      {!allFinished ? (
        <div className="status-strip" style={{ marginBottom: 16 }}>
          <div className="status-main">
            <span className="status-icon">
              <LoaderCircle size={18} className="spin" />
            </span>
            <div>
              <div className="status-title">正在检测 {rows.length} 条视频</div>
              <div className="status-stage">
                已完成 {finishedCount}/{rows.length} · 整体进度 {overallProgress}%
              </div>
            </div>
          </div>
          <div className="progress-track" aria-label="整体进度">
            <div className="progress-fill" style={{ width: `${overallProgress}%` }} />
          </div>
        </div>
      ) : null}

      <div className="batch-summary">
        <span className="badge badge-trust">
          <CheckCircle2 size={13} />
          可信 {verdictCounts.可信}
        </span>
        <span className="badge badge-suspect">
          <AlertTriangle size={13} />
          疑似生成合成 {verdictCounts.疑似生成合成}
        </span>
        <span className="badge badge-uncertain">无法确定 {verdictCounts.无法确定}</span>
        <span className="badge badge-neutral">检测失败 {failedCount}</span>
        {!allFinished ? <span className="badge badge-brand">检测中 {runningCount}</span> : null}
      </div>

      <section className="panel">
        <div className="panel-header">
          <h2 className="panel-title">鉴别结果汇总</h2>
          <span className="badge badge-neutral">{rows.length} 条</span>
        </div>
        <div className="batch-table-wrap">
          <table className="data-table batch-table">
            <thead>
              <tr>
                <th>#</th>
                <th>文件</th>
                <th>状态</th>
                <th>综合判定</th>
                <th>判定强度</th>
                <th>建议动作</th>
                <th>报告</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.id}>
                  <td className="muted">{row.index}</td>
                  <td style={{ maxWidth: 260 }}>
                    <div className="file-name">
                      {row.status?.original_filename ??
                        (row.lost ? "任务读取失败" : "读取中…")}
                    </div>
                    <div className="file-meta">
                      {row.status ? formatBytes(row.status.size_bytes) : row.id.slice(0, 8)}
                    </div>
                  </td>
                  <td>
                    <StatusCell status={row.status} lost={row.lost} />
                  </td>
                  <td>
                    {row.report ? (
                      <span className={`badge ${verdictBadgeClass(row.report.verdict)}`}>
                        {row.report.verdict}
                      </span>
                    ) : row.reportLost ? (
                      <span className="badge badge-uncertain">
                        <XCircle size={12} />
                        报告读取失败
                      </span>
                    ) : (
                      <span className="muted">—</span>
                    )}
                  </td>
                  <td>
                    {row.report ? (
                      `${(row.report.confidence * 100).toFixed(1)}%`
                    ) : (
                      <span className="muted">—</span>
                    )}
                  </td>
                  <td>
                    {row.report ? (
                      row.report.recommended_action
                    ) : (
                      <span className="muted">—</span>
                    )}
                  </td>
                  <td>
                    <Link className="table-link" href={`/jobs/${row.id}`}>
                      查看
                    </Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </>
  );
}

function StatusCell({ status, lost }: { status?: JobStatus; lost: boolean }) {
  if (lost) {
    return (
      <span className="badge badge-uncertain">
        <XCircle size={12} />
        读取失败
      </span>
    );
  }
  if (!status) {
    return (
      <span className="muted" style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
        <LoaderCircle size={13} className="spin" />
        读取中
      </span>
    );
  }
  if (status.status === "completed") {
    return (
      <span className="badge badge-trust">
        <CheckCircle2 size={12} />
        已完成
      </span>
    );
  }
  if (status.status === "failed") {
    return (
      <span
        className="badge badge-suspect"
        title={status.error_message ?? undefined}
      >
        <AlertTriangle size={12} />
        检测失败
      </span>
    );
  }
  return (
    <div className="batch-progress">
      <div className="progress-track">
        <div className="progress-fill" style={{ width: `${status.progress}%` }} />
      </div>
      <div className="file-meta" style={{ marginTop: 5 }}>
        {statusLabel(status.status)} · {status.stage} {status.progress}%
      </div>
    </div>
  );
}
