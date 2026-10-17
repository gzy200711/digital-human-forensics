"use client";

import {
  ArrowRight,
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  CircleX,
  Clock3,
  FileClock,
  FileVideo,
  ListFilter,
  LoaderCircle,
  RefreshCw,
  Search,
  ScanLine,
} from "lucide-react";
import Link from "next/link";
import { FormEvent, useEffect, useMemo, useState } from "react";

import { ApiError, formatBytes, listJobs } from "@/lib/api";
import type {
  JobHistoryItem,
  JobHistoryResponse,
  JobStatus,
  Verdict,
} from "@/types/report";

const PAGE_SIZE = 20;
const POLL_INTERVAL_MS = 5000;

type StatusFilter = "" | JobStatus["status"];
type VerdictFilter = "" | Verdict;

function parseTimestamp(value: string): Date {
  const hasTimezone = /(?:Z|[+-]\d{2}:\d{2})$/i.test(value);
  return new Date(hasTimezone ? value : `${value}Z`);
}

function formatDateTime(value: string | null): string {
  if (!value) return "—";
  const parsed = parseTimestamp(value);
  if (Number.isNaN(parsed.getTime())) return "—";
  return new Intl.DateTimeFormat("zh-CN", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  }).format(parsed);
}

function verdictBadgeClass(verdict: Verdict): string {
  if (verdict === "可信") return "badge badge-trust";
  if (verdict === "疑似生成合成") return "badge badge-suspect";
  return "badge badge-uncertain";
}

function statusDetail(item: JobHistoryItem): string {
  switch (item.status) {
    case "queued":
      return item.stage || "等待处理";
    case "processing":
      return `${item.progress}% · ${item.stage}`;
    case "failed":
      return item.error_code ? `错误码 ${item.error_code}` : "处理失败";
    case "completed":
      return item.report_ready ? "报告已生成" : "报告缺失";
  }
}

function StatusBadge({ item }: { item: JobHistoryItem }) {
  if (item.status === "processing") {
    return (
      <span className="badge badge-brand">
        <LoaderCircle size={13} className="spin" />
        检测中
      </span>
    );
  }
  if (item.status === "completed") {
    return (
      <span className="badge badge-neutral">
        <CheckCircle2 size={13} />
        已完成
      </span>
    );
  }
  if (item.status === "failed") {
    return (
      <span className="badge badge-suspect">
        <CircleX size={13} />
        失败
      </span>
    );
  }
  return (
    <span className="badge badge-neutral">
      <Clock3 size={13} />
      排队中
    </span>
  );
}

function HistorySkeleton() {
  return (
    <div className="history-skeleton" aria-label="正在读取检测历史">
      {Array.from({ length: 5 }, (_, index) => (
        <div className="history-skeleton-row" key={index}>
          <span className="skeleton" style={{ width: "72%", height: 18 }} />
          <span className="skeleton" style={{ width: 76, height: 22 }} />
          <span className="skeleton" style={{ width: 92, height: 22 }} />
          <span className="skeleton" style={{ width: 54, height: 18 }} />
          <span className="skeleton" style={{ width: "84%", height: 16 }} />
        </div>
      ))}
    </div>
  );
}

export function HistoryView() {
  const [queryInput, setQueryInput] = useState("");
  const [query, setQuery] = useState("");
  const [status, setStatus] = useState<StatusFilter>("");
  const [verdict, setVerdict] = useState<VerdictFilter>("");
  const [offset, setOffset] = useState(0);
  const [refreshKey, setRefreshKey] = useState(0);
  const [data, setData] = useState<JobHistoryResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const hasFilters = Boolean(query || status || verdict);
  const liveCount = useMemo(
    () =>
      data?.items.filter(
        (item) => item.status === "queued" || item.status === "processing",
      ).length ?? 0,
    [data],
  );

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError(null);
    listJobs(
      {
        query,
        status: status || undefined,
        verdict: verdict || undefined,
        limit: PAGE_SIZE,
        offset,
      },
      controller.signal,
    )
      .then((result) => {
        setData(result);
        setLoading(false);
      })
      .catch((caught: unknown) => {
        if (controller.signal.aborted) return;
        setError(
          caught instanceof ApiError
            ? caught.message
            : "检测历史读取失败，请检查服务状态后重试。",
        );
        setLoading(false);
      });
    return () => controller.abort();
  }, [query, status, verdict, offset, refreshKey]);

  useEffect(() => {
    if (liveCount === 0) return;
    const timer = window.setInterval(() => {
      setRefreshKey((value) => value + 1);
    }, POLL_INTERVAL_MS);
    return () => window.clearInterval(timer);
  }, [liveCount]);

  function applySearch(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setOffset(0);
    setQuery(queryInput.trim());
  }

  function clearFilters() {
    setQueryInput("");
    setQuery("");
    setStatus("");
    setVerdict("");
    setOffset(0);
  }

  function retry() {
    setRefreshKey((value) => value + 1);
  }

  const initialLoading = loading && data === null;
  const pageStart = data && data.items.length > 0 ? data.offset + 1 : 0;
  const pageEnd = data ? data.offset + data.items.length : 0;
  const pageNumber = Math.floor(offset / PAGE_SIZE) + 1;
  const pageCount = Math.max(1, Math.ceil((data?.total ?? 0) / PAGE_SIZE));

  return (
    <>
      <div className="page-header">
        <div>
          <div className="page-eyebrow">Detection Archive</div>
          <h1 className="page-title">历史检测</h1>
          <p className="page-subtitle">
            查看提交过的检测任务、当前处理状态与最终结论，并进入原报告复核证据。
          </p>
        </div>
        <Link className="button button-primary" href="/">
          <ScanLine size={15} />
          新的检测
        </Link>
      </div>

      <section className="panel" style={{ marginBottom: 18 }}>
        <div className="panel-header">
          <h2 className="panel-title">筛选记录</h2>
          <span className="badge badge-neutral">
            <ListFilter size={13} />
            {hasFilters ? "已应用筛选" : "全部记录"}
          </span>
        </div>
        <div className="panel-body">
          <form className="history-filter-form" onSubmit={applySearch}>
            <div className="history-search">
              <Search size={17} aria-hidden="true" />
              <input
                className="history-search-input"
                value={queryInput}
                onChange={(event) => setQueryInput(event.target.value)}
                placeholder="搜索文件名或任务 ID"
                aria-label="搜索文件名或任务 ID"
                maxLength={200}
              />
              <button className="button button-primary" type="submit" disabled={loading}>
                查询
              </button>
            </div>
            <div className="history-selects">
              <label className="history-select">
                <span>处理状态</span>
                <select
                  className="select"
                  value={status}
                  onChange={(event) => {
                    setStatus(event.target.value as StatusFilter);
                    setOffset(0);
                  }}
                >
                  <option value="">全部状态</option>
                  <option value="queued">排队中</option>
                  <option value="processing">检测中</option>
                  <option value="completed">已完成</option>
                  <option value="failed">失败</option>
                </select>
              </label>
              <label className="history-select">
                <span>综合判定</span>
                <select
                  className="select"
                  value={verdict}
                  onChange={(event) => {
                    setVerdict(event.target.value as VerdictFilter);
                    setOffset(0);
                  }}
                >
                  <option value="">全部结论</option>
                  <option value="可信">可信</option>
                  <option value="无法确定">无法确定</option>
                  <option value="疑似生成合成">疑似生成合成</option>
                </select>
              </label>
              <button
                className="button button-secondary history-reset"
                type="button"
                onClick={clearFilters}
                disabled={!hasFilters && !queryInput}
              >
                重置
              </button>
            </div>
          </form>
        </div>
      </section>

      <section className="panel" aria-busy={loading}>
        <div className="panel-header">
          <div>
            <h2 className="panel-title">检测记录</h2>
            <div className="history-result-line" aria-live="polite">
              {data
                ? data.total > 0
                  ? `共 ${data.total} 条，当前显示 ${pageStart}–${pageEnd} 条`
                  : "没有找到检测记录"
                : error
                  ? "检测记录读取失败"
                  : "正在读取检测记录"}
              {liveCount > 0 ? ` · ${liveCount} 条正在处理，每 5 秒自动刷新` : ""}
            </div>
          </div>
          <button
            className="icon-button"
            type="button"
            aria-label="刷新检测记录"
            title="刷新检测记录"
            onClick={retry}
            disabled={loading}
          >
            <RefreshCw size={16} className={loading && data ? "spin" : undefined} />
          </button>
        </div>

        {error ? (
          <div className="history-error">
            <div className="alert alert-error" role="alert">
              {error}
            </div>
            <button className="button button-secondary" type="button" onClick={retry}>
              <RefreshCw size={15} />
              重试
            </button>
          </div>
        ) : null}

        {initialLoading ? (
          <HistorySkeleton />
        ) : data && data.items.length > 0 ? (
          <>
            <div className="history-table-head" aria-hidden="true">
              <span>检测文件</span>
              <span>状态</span>
              <span>综合判定</span>
              <span>判定强度</span>
              <span>提交时间</span>
              <span>报告</span>
            </div>
            <div className="history-list" data-loading={loading}>
              {data.items.map((item) => (
                <article className="history-row" key={item.job_id}>
                  <div className="history-file">
                    <span className="history-file-icon">
                      <FileVideo size={19} />
                    </span>
                    <div className="history-file-copy">
                      <Link
                        className="history-file-name"
                        href={`/jobs/${encodeURIComponent(item.job_id)}`}
                        title={item.original_filename}
                      >
                        {item.original_filename}
                      </Link>
                      <div className="history-file-meta">
                        <span>{item.source_type === "url" ? "媒体 URL" : "上传文件"}</span>
                        <span>{formatBytes(item.size_bytes)}</span>
                        <span className="inline-code">{item.job_id.slice(0, 8)}</span>
                      </div>
                    </div>
                  </div>

                  <div className="history-cell">
                    <span className="history-cell-label">状态</span>
                    <StatusBadge item={item} />
                    <span
                      className="history-cell-detail"
                      title={
                        item.status === "failed" ? item.error_message ?? undefined : undefined
                      }
                    >
                      {statusDetail(item)}
                    </span>
                  </div>

                  <div className="history-cell">
                    <span className="history-cell-label">综合判定</span>
                    {item.verdict && item.report_ready ? (
                      <span className={verdictBadgeClass(item.verdict)}>{item.verdict}</span>
                    ) : (
                      <span className="history-empty-value">—</span>
                    )}
                    {item.recommended_action ? (
                      <span className="history-cell-detail">{item.recommended_action}</span>
                    ) : null}
                  </div>

                  <div className="history-cell">
                    <span className="history-cell-label">判定强度</span>
                    <span className="history-score">
                      {item.confidence === null
                        ? "—"
                        : `${(Math.max(0, Math.min(1, item.confidence)) * 100).toFixed(1)}%`}
                    </span>
                  </div>

                  <div className="history-cell">
                    <span className="history-cell-label">提交时间</span>
                    <time className="history-date" dateTime={item.created_at}>
                      {formatDateTime(item.created_at)}
                    </time>
                  </div>

                  <div className="history-action">
                    <Link
                      className="button button-secondary"
                      href={`/jobs/${encodeURIComponent(item.job_id)}`}
                    >
                      {item.report_ready ? "查看报告" : "查看进度"}
                      <ArrowRight size={14} />
                    </Link>
                  </div>
                </article>
              ))}
            </div>

            {data.total > PAGE_SIZE ? (
              <div className="history-pagination">
                <span>
                  第 {pageNumber} / {pageCount} 页
                </span>
                <div className="history-pagination-actions">
                  <button
                    className="button button-secondary"
                    type="button"
                    onClick={() => setOffset(Math.max(0, offset - PAGE_SIZE))}
                    disabled={offset === 0 || loading}
                  >
                    <ChevronLeft size={15} />
                    上一页
                  </button>
                  <button
                    className="button button-secondary"
                    type="button"
                    onClick={() => setOffset(offset + PAGE_SIZE)}
                    disabled={offset + data.items.length >= data.total || loading}
                  >
                    下一页
                    <ChevronRight size={15} />
                  </button>
                </div>
              </div>
            ) : null}
          </>
        ) : data ? (
          <div className="empty-state">
            <div>
              <span className="empty-icon">
                <FileClock size={24} />
              </span>
              <div style={{ fontWeight: 700 }}>
                {hasFilters ? "没有符合筛选条件的记录" : "还没有检测记录"}
              </div>
              <div className="muted" style={{ marginTop: 7 }}>
                {hasFilters
                  ? "调整关键词、处理状态或综合判定后再试。"
                  : "完成第一次检测后，任务、结论和报告入口会显示在这里。"}
              </div>
              <div style={{ marginTop: 16 }}>
                {hasFilters ? (
                  <button className="button button-secondary" type="button" onClick={clearFilters}>
                    <ListFilter size={15} />
                    清除筛选
                  </button>
                ) : (
                  <Link className="button button-primary" href="/">
                    <ScanLine size={15} />
                    发起检测
                  </Link>
                )}
              </div>
            </div>
          </div>
        ) : null}
      </section>
    </>
  );
}
