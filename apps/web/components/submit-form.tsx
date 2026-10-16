"use client";

import {
  ArrowRight,
  CircleAlert,
  FileVideo,
  Link2,
  ShieldCheck,
  UploadCloud,
  X,
} from "lucide-react";
import { useRouter } from "next/navigation";
import { DragEvent, FormEvent, useRef, useState } from "react";

import { ApiError, MAX_BATCH_JOBS, createJob, formatBytes } from "@/lib/api";

type SourceMode = "file" | "url";

const MAX_BYTES = 500 * 1024 * 1024;
const SUBMIT_CONCURRENCY = 3;

interface SubmitItem {
  name: string;
  file?: File;
  sourceUrl?: string;
}

type SubmitOutcome =
  | { ok: true; jobId: string }
  | { ok: false; name: string; message: string };

/** 以固定并发数遍历执行任务，worker 内部自行捕获异常。 */
async function runPool<T>(
  items: T[],
  limit: number,
  worker: (item: T, index: number) => Promise<void>,
): Promise<void> {
  let cursor = 0;
  async function next(): Promise<void> {
    const index = cursor;
    cursor += 1;
    if (index >= items.length) return;
    await worker(items[index], index);
    await next();
  }
  const lanes = Math.min(limit, items.length);
  await Promise.all(Array.from({ length: lanes }, () => next()));
}

export function SubmitForm() {
  const router = useRouter();
  const inputRef = useRef<HTMLInputElement>(null);
  const filesRef = useRef<File[]>([]);
  const [mode, setMode] = useState<SourceMode>("file");
  const [files, setFiles] = useState<File[]>([]);
  const [sourceUrls, setSourceUrls] = useState("");
  const [retentionDays, setRetentionDays] = useState(7);
  const [consentAck, setConsentAck] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [submitProgress, setSubmitProgress] = useState(0);
  const [submitPercent, setSubmitPercent] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  const urlCount = sourceUrls
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean).length;
  const totalCount = mode === "file" ? files.length : urlCount;

  function updateFiles(next: File[]) {
    filesRef.current = next;
    setFiles(next);
  }

  function selectFiles(incoming: File[] | undefined) {
    setError(null);
    if (!incoming || incoming.length === 0) return;
    const problems: string[] = [];
    const merged = [...filesRef.current];
    let oversize = 0;
    for (const candidate of incoming) {
      if (candidate.size > MAX_BYTES) {
        oversize += 1;
        continue;
      }
      const duplicate = merged.some(
        (item) => item.name === candidate.name && item.size === candidate.size,
      );
      if (!duplicate) merged.push(candidate);
    }
    let next = merged;
    if (merged.length > MAX_BATCH_JOBS) {
      next = merged.slice(0, MAX_BATCH_JOBS);
      problems.push(
        `一次最多检测 ${MAX_BATCH_JOBS} 个文件，超出的 ${merged.length - MAX_BATCH_JOBS} 个已被忽略`,
      );
    }
    if (oversize > 0) problems.push(`${oversize} 个文件超过 500 MB 限制，已忽略`);
    if (problems.length > 0) setError(`${problems.join("；")}。`);
    updateFiles(next);
  }

  function removeFile(index: number) {
    setError(null);
    updateFiles(filesRef.current.filter((_, itemIndex) => itemIndex !== index));
  }

  function handleDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setDragging(false);
    selectFiles(Array.from(event.dataTransfer.files ?? []));
  }

  function parseUrls(): string[] | null {
    const lines = sourceUrls
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter(Boolean);
    const unique = [...new Set(lines)];
    if (unique.length === 0) {
      setError("请输入至少一条可访问的媒体 URL。");
      return null;
    }
    if (unique.length > MAX_BATCH_JOBS) {
      setError(`一次最多检测 ${MAX_BATCH_JOBS} 条 URL，当前为 ${unique.length} 条。`);
      return null;
    }
    const invalidIndex = unique.findIndex((url) => !/^https?:\/\/\S+$/i.test(url));
    if (invalidIndex >= 0) {
      setError(`第 ${invalidIndex + 1} 条链接格式无效：${unique[invalidIndex].slice(0, 80)}`);
      return null;
    }
    return unique;
  }

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    setError(null);
    if (!consentAck) {
      setError("请先确认拥有素材处理权限。");
      return;
    }
    let items: SubmitItem[];
    if (mode === "file") {
      if (files.length === 0) {
        setError("请选择待检测的视频或音频文件。");
        return;
      }
      items = files.map((file) => ({ name: file.name, file }));
    } else {
      const urls = parseUrls();
      if (!urls) return;
      items = urls.map((url) => ({ name: url, sourceUrl: url }));
    }

    setSubmitting(true);
    setSubmitProgress(0);
    const sizes = items.map((item) => item.file?.size ?? 0);
    const totalBytes = sizes.reduce((sum, size) => sum + size, 0);
    const loadedBytes = new Array<number>(items.length).fill(0);
    setSubmitPercent(totalBytes > 0 ? 0 : null);
    let lastPercent = -1;
    const reportProgress = (index: number, value: number) => {
      if (totalBytes <= 0) return;
      loadedBytes[index] = Math.min(value, sizes[index]);
      const sum = loadedBytes.reduce((total, loaded) => total + loaded, 0);
      const percent = Math.min(100, Math.round((sum / totalBytes) * 100));
      if (percent !== lastPercent) {
        lastPercent = percent;
        setSubmitPercent(percent);
      }
    };
    const outcomes: SubmitOutcome[] = new Array(items.length);
    await runPool(items, SUBMIT_CONCURRENCY, async (item, index) => {
      const submitOnce = async () => {
        const result = await createJob(
          {
            file: item.file,
            sourceUrl: item.sourceUrl,
            retentionDays,
            consentAck,
          },
          (loaded) => {
            loadedBytes[index] = loaded;
            reportProgress(index, loaded);
          },
        );
        reportProgress(index, sizes[index]);
        return result;
      };
      try {
        let result: { job_id: string } | null = null;
        let lastError: unknown = null;
        for (let attempt = 0; attempt < 3 && result === null; attempt += 1) {
          try {
            result = await submitOnce();
          } catch (caught) {
            lastError = caught;
            const retryable = caught instanceof ApiError && caught.code === "NETWORK_ERROR";
            if (!retryable || attempt === 2) throw caught;
            // 链路闪断：等待后自动重试（共 3 次机会），避免用户手动重传。
            await new Promise((resolve) => setTimeout(resolve, attempt === 0 ? 2500 : 6000));
            loadedBytes[index] = 0;
          }
        }
        if (result === null) throw lastError ?? new Error("提交失败");
        outcomes[index] = { ok: true, jobId: result.job_id };
      } catch (caught) {
        outcomes[index] = {
          ok: false,
          name: item.name,
          message:
            caught instanceof ApiError
              ? `${caught.code}: ${caught.message}`
              : "提交失败，请检查服务状态。",
        };
      } finally {
        setSubmitProgress((done) => done + 1);
      }
    });

    const succeeded = outcomes
      .filter((outcome): outcome is Extract<SubmitOutcome, { ok: true }> => outcome?.ok === true)
      .map((outcome) => outcome.jobId);
    const failed = outcomes.filter(
      (outcome): outcome is Extract<SubmitOutcome, { ok: false }> => Boolean(outcome) && !outcome.ok,
    );

    if (succeeded.length === 0) {
      setError(`全部提交失败：${failed[0]?.message ?? "未知错误"}`);
      setSubmitting(false);
      return;
    }
    if (succeeded.length === 1 && failed.length === 0) {
      router.push(`/jobs/${succeeded[0]}`);
      return;
    }
    const params = new URLSearchParams();
    params.set("ids", succeeded.join(","));
    if (failed.length > 0) {
      params.set("failed", failed.map((item) => encodeURIComponent(item.name)).join("|"));
    }
    router.push(`/batch?${params.toString()}`);
  }

  return (
    <form className="panel" onSubmit={handleSubmit}>
      <div className="panel-header">
        <h2 className="panel-title">提交检测</h2>
        <div className="segmented" aria-label="输入方式">
          <button
            className="segment"
            data-active={mode === "file"}
            onClick={() => setMode("file")}
            type="button"
          >
            <UploadCloud size={15} />
            上传文件
          </button>
          <button
            className="segment"
            data-active={mode === "url"}
            onClick={() => setMode("url")}
            type="button"
          >
            <Link2 size={15} />
            媒体 URL
          </button>
        </div>
      </div>
      <div className="panel-body">
        {mode === "file" ? (
          <div
            className="dropzone"
            data-dragging={dragging}
            onClick={() => inputRef.current?.click()}
            onDragEnter={(event) => {
              event.preventDefault();
              setDragging(true);
            }}
            onDragOver={(event) => event.preventDefault()}
            onDragLeave={() => setDragging(false)}
            onDrop={handleDrop}
            role="button"
            tabIndex={0}
            onKeyDown={(event) => {
              if (event.key === "Enter" || event.key === " ") {
                inputRef.current?.click();
              }
            }}
          >
            <input
              ref={inputRef}
              type="file"
              hidden
              multiple
              accept="video/*,audio/*"
              onChange={(event) => {
                selectFiles(event.target.files ? Array.from(event.target.files) : undefined);
                event.target.value = "";
              }}
            />
            {files.length > 0 ? (
              <div style={{ width: "100%" }}>
                <div className="file-list">
                  {files.map((item, index) => (
                    <div className="selected-file" key={`${item.name}-${item.size}-${index}`}>
                      <span className="selected-file-icon">
                        <FileVideo size={22} />
                      </span>
                      <div style={{ minWidth: 0, flex: 1 }}>
                        <div className="file-name">{item.name}</div>
                        <div className="file-meta">
                          {formatBytes(item.size)} · {item.type || "未知类型"}
                        </div>
                      </div>
                      <button
                        className="icon-button"
                        type="button"
                        aria-label={`移除 ${item.name}`}
                        disabled={submitting}
                        onClick={(event) => {
                          event.stopPropagation();
                          removeFile(index);
                        }}
                      >
                        <X size={16} />
                      </button>
                    </div>
                  ))}
                </div>
                <div className="file-list-footer">
                  <span>
                    共 {files.length} / {MAX_BATCH_JOBS} 个文件 · 合计{" "}
                    {formatBytes(files.reduce((sum, item) => sum + item.size, 0))}
                  </span>
                  <button
                    className="file-list-clear"
                    type="button"
                    disabled={submitting}
                    onClick={(event) => {
                      event.stopPropagation();
                      setError(null);
                      updateFiles([]);
                    }}
                  >
                    清空列表
                  </button>
                </div>
              </div>
            ) : (
              <div>
                <span className="dropzone-icon">
                  <UploadCloud size={26} />
                </span>
                <div className="dropzone-title">拖入媒体文件</div>
                <div className="dropzone-caption">
                  MP4 / MOV / MKV / WebM / AVI / WAV / MP3 / M4A / FLAC
                  <br />
                  单个最大 500 MB，一次最多 {MAX_BATCH_JOBS} 个，可批量拖入
                </div>
              </div>
            )}
          </div>
        ) : (
          <div className="field" style={{ minHeight: 246, alignContent: "center" }}>
            <label htmlFor="source-urls">媒体 URL（每行一条，最多 {MAX_BATCH_JOBS} 条）</label>
            <textarea
              id="source-urls"
              className="textarea"
              placeholder={"https://example.com/video-1.mp4\nhttps://example.com/video-2.mp4"}
              value={sourceUrls}
              onChange={(event) => setSourceUrls(event.target.value)}
            />
            <div className="muted" style={{ fontSize: 12, lineHeight: 1.6 }}>
              每条 URL 创建一个检测任务；服务端会限制内网地址、重定向次数、文件大小和下载超时。
            </div>
          </div>
        )}

        <div className="form-row">
          <div className="field">
            <label htmlFor="retention">原始文件留存</label>
            <select
              id="retention"
              className="select"
              value={retentionDays}
              onChange={(event) => setRetentionDays(Number(event.target.value))}
            >
              <option value={1}>1 天</option>
              <option value={7}>7 天</option>
              <option value={30}>30 天</option>
            </select>
          </div>
          <span className="badge badge-neutral">
            <ShieldCheck size={13} />
            最小化留存
          </span>
        </div>

        <label className="checkbox-row">
          <input
            type="checkbox"
            checked={consentAck}
            onChange={(event) => setConsentAck(event.target.checked)}
          />
          <span>
            我确认拥有处理该素材的合法权限，并理解本平台输出的是技术检测证据，不构成司法鉴定结论。
          </span>
        </label>

        {error ? (
          <div className="alert alert-error" role="alert" style={{ marginBottom: 14 }}>
            <CircleAlert size={15} style={{ verticalAlign: "-2px", marginRight: 6 }} />
            {error}
          </div>
        ) : null}

        <div className="actions">
          {submitting ? (
            <div className="submit-progress">
              {submitPercent !== null ? (
                <div className="progress-track">
                  <div className="progress-fill" style={{ width: `${submitPercent}%` }} />
                </div>
              ) : null}
              <div className="file-meta" style={{ marginTop: 6 }}>
                {submitPercent !== null
                  ? `上传进度 ${submitPercent}%（已完成提交 ${submitProgress}/${totalCount}）`
                  : `正在提交 ${submitProgress}/${totalCount}…`}
              </div>
            </div>
          ) : null}
          <button className="button button-primary" disabled={submitting} type="submit">
            {submitting
              ? "正在提交…"
              : totalCount > 1
                ? `开始检测（${totalCount} 个）`
                : "开始检测"}
            <ArrowRight size={16} />
          </button>
        </div>
      </div>
    </form>
  );
}
