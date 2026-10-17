import type {
  ForensicReport,
  JobHistoryResponse,
  JobStatus,
  Verdict,
} from "@/types/report";

export class ApiError extends Error {
  status: number;
  code: string;

  constructor(message: string, status: number, code: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
  }
}

async function parseError(response: Response): Promise<ApiError> {
  let payload: {
    detail?: { error?: string; message?: string };
    error?: string;
    message?: string;
  } = {};
  let responseText = "";
  try {
    responseText = await response.text();
    payload = JSON.parse(responseText) as typeof payload;
  } catch {
    // 非 JSON 错误由下面的默认文案处理。
  }
  const detail = payload.detail ?? payload;
  const fallback =
    response.status === 413
      ? "上传文件过大，请选择不超过 500 MB 的媒体文件。"
      : response.status === 502 || response.status === 504
        ? "检测服务暂时不可用，请稍后重试。"
        : response.status === 500
          ? "服务处理请求时发生错误，请稍后重试。"
          : responseText && response.headers.get("content-type")?.includes("text/")
            ? responseText.replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim().slice(0, 180)
            : "请求失败，请稍后重试。";
  return new ApiError(
    detail.message ?? fallback,
    response.status,
    detail.error ?? "REQUEST_FAILED",
  );
}

export function createJob(
  input: {
    file?: File;
    sourceUrl?: string;
    retentionDays: number;
    consentAck: boolean;
  },
  onProgress?: (loaded: number, total: number) => void,
): Promise<{ job_id: string }> {
  const form = new FormData();
  if (input.file) form.append("file", input.file);
  if (input.sourceUrl) form.append("source_url", input.sourceUrl);
  form.append("retention_days", String(input.retentionDays));
  form.append("consent_ack", String(input.consentAck));

  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/v1/jobs");
    if (onProgress) {
      xhr.upload.onprogress = (event) => {
        if (event.lengthComputable) onProgress(event.loaded, event.total);
      };
    }
    xhr.onerror = () =>
      reject(new ApiError("网络错误，上传失败，请重试。", 0, "NETWORK_ERROR"));
    xhr.onabort = () => reject(new ApiError("上传已取消。", 0, "UPLOAD_ABORTED"));
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(JSON.parse(xhr.responseText) as { job_id: string });
        } catch {
          reject(new ApiError("服务返回了无法解析的响应。", xhr.status, "INVALID_RESPONSE"));
        }
        return;
      }
      let payload: {
        detail?: { error?: string; message?: string };
        error?: string;
        message?: string;
      } = {};
      try {
        payload = JSON.parse(xhr.responseText) as typeof payload;
      } catch {
        // 非 JSON 响应走默认文案。
      }
      const detail = payload.detail ?? payload;
      const fallback =
        xhr.status === 413
          ? "上传文件过大，请选择不超过 500 MB 的媒体文件。"
          : xhr.status === 502 || xhr.status === 504
            ? "检测服务暂时不可用，请稍后重试。"
            : "提交失败，请稍后重试。";
      reject(
        new ApiError(detail.message ?? fallback, xhr.status, detail.error ?? "REQUEST_FAILED"),
      );
    };
    xhr.send(form);
  });
}

export async function getJob(jobId: string): Promise<JobStatus> {
  const response = await fetch(`/api/v1/jobs/${encodeURIComponent(jobId)}`, {
    cache: "no-store",
  });
  if (!response.ok) throw await parseError(response);
  return response.json() as Promise<JobStatus>;
}

export async function listJobs(
  input: {
    query?: string;
    status?: JobStatus["status"];
    verdict?: Verdict;
    limit?: number;
    offset?: number;
  } = {},
  signal?: AbortSignal,
): Promise<JobHistoryResponse> {
  const params = new URLSearchParams();
  if (input.query?.trim()) params.set("q", input.query.trim());
  if (input.status) params.set("status", input.status);
  if (input.verdict) params.set("verdict", input.verdict);
  params.set("limit", String(input.limit ?? 20));
  params.set("offset", String(input.offset ?? 0));
  const response = await fetch(`/api/v1/jobs?${params.toString()}`, {
    cache: "no-store",
    signal,
  });
  if (!response.ok) throw await parseError(response);
  return response.json() as Promise<JobHistoryResponse>;
}

export async function getReport(jobId: string): Promise<ForensicReport> {
  const response = await fetch(
    `/api/v1/jobs/${encodeURIComponent(jobId)}/report`,
    { cache: "no-store" },
  );
  if (!response.ok) throw await parseError(response);
  return response.json() as Promise<ForensicReport>;
}

export async function submitAppeal(
  jobId: string,
  input: { reason: string; contact?: string },
): Promise<{ appeal_id: string; status: string }> {
  const response = await fetch(
    `/api/v1/jobs/${encodeURIComponent(jobId)}/appeal`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(input),
    },
  );
  if (!response.ok) throw await parseError(response);
  return response.json() as Promise<{ appeal_id: string; status: string }>;
}

export function formatBytes(value: number | null): string {
  if (value === null || !Number.isFinite(value)) return "未知";
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
  if (value < 1024 * 1024 * 1024) {
    return `${(value / 1024 / 1024).toFixed(1)} MB`;
  }
  return `${(value / 1024 / 1024 / 1024).toFixed(2)} GB`;
}

export function formatDuration(value: number | null): string {
  if (value === null || !Number.isFinite(value)) return "未知";
  const totalSeconds = Math.max(0, Math.round(value));
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return minutes > 0
    ? `${minutes}分${String(seconds).padStart(2, "0")}秒`
    : `${seconds}秒`;
}

export function verdictTone(verdict: ForensicReport["verdict"]): string {
  if (verdict === "可信") return "trusted";
  if (verdict === "疑似生成合成") return "suspected";
  return "uncertain";
}

/** 单次批量检测的任务数上限。 */
export const MAX_BATCH_JOBS = 40;
