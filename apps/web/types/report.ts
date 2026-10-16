export type Verdict = "可信" | "无法确定" | "疑似生成合成";

export type ModalityKey = "visual" | "audio" | "cross_modal" | "provenance";

export type ModalityStatus = "available" | "partial" | "unavailable";

export type Direction = "指向生成" | "指向真实" | "中性";

export type Strength = "弱" | "中" | "强";

export interface TimeRange {
  start: number;
  end: number;
  score: number;
  reason: string;
}

export interface EvidenceRegion {
  label: string;
  x?: number | null;
  y?: number | null;
  width?: number | null;
  height?: number | null;
}

export interface EvidenceModel {
  name: string;
  version: string;
  license: string | null;
}

export interface EvidenceItem {
  id: string;
  signal: string;
  display_name: string;
  layer?: string | null;
  source?: string | null;
  direction: Direction;
  strength: Strength;
  score: number | null;
  unit?: string | null;
  raw_value: Record<string, unknown>;
  explanation: string;
  time_range: TimeRange | null;
  region?: EvidenceRegion | null;
  model?: EvidenceModel | null;
  uncertainty?: string | null;
  limitations?: string[];
  available: boolean;
}

export interface ModalityVerdict {
  modality: ModalityKey;
  label: string;
  verdict: Verdict;
  score: number | null;
  coverage: number | null;
  status: ModalityStatus;
  signal_ids: string[];
  summary: string;
  limitations: string[];
}

export interface ModalityCoverage {
  full_frame: number | null;
  face: number | null;
  audio: number | null;
  cross_modal: number | null;
  notes: string[];
}

export interface GrayZone {
  minimum: number;
  maximum: number;
  score: number | null;
  in_gray_zone: boolean;
  note: string;
}

export interface CalibratedProbability {
  value: number | null;
  is_calibrated: boolean;
  calibration_version: string;
  calibration_group: string;
  distribution: string;
  confidence_interval: [number, number] | null;
  semantics: string;
}

export interface ThresholdPolicy {
  policy_id: string;
  source: string;
  gray_zone: [number, number];
  target_fprs: number[];
  real_negative_samples: number | null;
  fpr_evidence_status: "verified" | "insufficient" | "unknown";
  statement: string;
}

export interface ModelSnapshot {
  registry_version: string;
  captured_at: string;
  is_research_only: boolean;
  note: string;
}

export interface AssessabilityFactor {
  name: string;
  value: number | string | boolean | null;
  status: "good" | "warning" | "bad" | "unknown";
  explanation: string;
}

export interface ReviewStatement {
  claim: string;
  evidence_ids: string[];
  kind: "生成风险" | "真实性提示" | "不确定性" | "质量限制";
}

export interface EvidenceReview {
  status: "not_triggered" | "completed" | "unavailable";
  triggered: boolean;
  provider: string;
  model_version: string | null;
  license: string | null;
  summary: string;
  statements: ReviewStatement[];
  rejected_claim_count: number;
  note: string;
}

export interface ForensicReport {
  schema_version: string;
  report_id: string;
  job_id: string;
  generated_at: string;
  media: {
    filename: string;
    mime_type: string | null;
    size_bytes: number | null;
    sha256: string | null;
    duration_seconds: number | null;
    width: number | null;
    height: number | null;
    frame_rate: number | null;
    video_codec: string | null;
    audio_codec: string | null;
    has_audio: boolean;
  };
  verdict: Verdict;
  confidence: number;
  calibrated_score: number;
  calibrated_probability?: CalibratedProbability | null;
  summary: string;
  modality_verdicts?: ModalityVerdict[];
  coverage?: ModalityCoverage;
  gray_zone?: GrayZone | null;
  abstention_reasons?: string[];
  abstention_codes?: string[];
  time_ranges: TimeRange[];
  evidence: EvidenceItem[];
  review?: EvidenceReview | null;
  assessability: {
    score: number;
    status: "可鉴定" | "勉强可鉴定" | "不可鉴定";
    face_detected_ratio: number | null;
    factors: AssessabilityFactor[];
    reasons: string[];
  };
  provenance: {
    checks: Array<{
      name: string;
      status: "阳性" | "阴性" | "未检查" | "不可用";
      positive: boolean;
      details: Record<string, unknown>;
      explanation: string;
    }>;
    explicit_ai_indicators: string[];
    status?: "阳性" | "无声明" | "未检查";
    note: string;
    limitation?: string;
  };
  limitations: string[];
  recommended_action: "通过" | "人工复核" | "拦截并允许申诉";
  artifacts: {
    media_url: string | null;
    heatmap_url: string | null;
    keyframe_url: string | null;
    json_report_url: string | null;
    html_report_url: string | null;
  };
  model_versions: Array<{
    name: string;
    version: string;
    license: string | null;
  }>;
  model_snapshot?: ModelSnapshot | null;
  threshold_policy?: ThresholdPolicy | null;
  audit_trace_id?: string | null;
  processing: Record<string, unknown>;
}

export interface JobStatus {
  job_id: string;
  status: "queued" | "processing" | "completed" | "failed";
  progress: number;
  stage: string;
  original_filename: string;
  mime_type: string | null;
  size_bytes: number | null;
  error_code: string | null;
  error_message: string | null;
  created_at: string;
  updated_at: string;
  started_at: string | null;
  completed_at: string | null;
  expires_at: string | null;
  report_ready: boolean;
}
