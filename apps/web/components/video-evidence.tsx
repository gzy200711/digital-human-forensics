"use client";

import { Eye, EyeOff, ScanLine } from "lucide-react";
import { MouseEvent, useRef, useState } from "react";

import type { ForensicReport, TimeRange } from "@/types/report";

export function VideoEvidence({ report }: { report: ForensicReport }) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [showHeatmap, setShowHeatmap] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(report.media.duration_seconds ?? 0);

  function seekFromEvent(event: MouseEvent<HTMLDivElement>) {
    if (!videoRef.current || duration <= 0) return;
    const bounds = event.currentTarget.getBoundingClientRect();
    const ratio = Math.min(
      1,
      Math.max(0, (event.clientX - bounds.left) / bounds.width),
    );
    videoRef.current.currentTime = ratio * duration;
  }

  return (
    <section className="panel">
      <div className="panel-header">
        <h2 className="panel-title">媒体与时间定位</h2>
        <button
          className="button button-secondary"
          type="button"
          disabled={!report.artifacts.heatmap_url}
          onClick={() => setShowHeatmap((current) => !current)}
        >
          {showHeatmap ? <EyeOff size={15} /> : <Eye size={15} />}
          {showHeatmap ? "隐藏热力图" : "显示热力图"}
        </button>
      </div>
      <div className="video-shell">
        <div className="video-stage">
          {report.artifacts.media_url ? (
            <video
              ref={videoRef}
              controls
              preload="metadata"
              src={report.artifacts.media_url}
              onLoadedMetadata={(event) => {
                const value = event.currentTarget.duration;
                setDuration(Number.isFinite(value) ? value : duration);
              }}
              onTimeUpdate={(event) => setCurrentTime(event.currentTarget.currentTime)}
            />
          ) : (
            <div className="empty-state">
              <div>
                <span className="empty-icon">
                  <ScanLine size={24} />
                </span>
                <div>媒体预览不可用</div>
              </div>
            </div>
          )}
          {showHeatmap && report.artifacts.heatmap_url ? (
            <img
              className="heatmap-overlay"
              src={report.artifacts.heatmap_url}
              alt="人脸区域证据热力图"
            />
          ) : null}
        </div>
        <div
          className="timeline"
          onClick={seekFromEvent}
          role="slider"
          aria-label="检测时间线"
          aria-valuemin={0}
          aria-valuemax={Math.round(duration)}
          aria-valuenow={Math.round(currentTime)}
          tabIndex={0}
        >
          {report.time_ranges.map((range, index) => (
            <TimelineMarker
              key={`${range.start}-${range.end}-${index}`}
              range={range}
              duration={duration}
            />
          ))}
          {duration > 0 ? (
            <span
              className="timeline-head"
              style={{ left: `${Math.min(100, (currentTime / duration) * 100)}%` }}
            />
          ) : null}
        </div>
        <div className="video-toolbar">
          <span>
            {report.time_ranges.length > 0
              ? `${report.time_ranges.length} 个疑似区间`
              : "未形成时间定位"}
          </span>
          <span>
            {formatClock(currentTime)} / {formatClock(duration)}
          </span>
        </div>
      </div>
    </section>
  );
}

function TimelineMarker({
  range,
  duration,
}: {
  range: TimeRange;
  duration: number;
}) {
  if (duration <= 0) return null;
  const left = Math.max(0, Math.min(100, (range.start / duration) * 100));
  const width = Math.max(
    0.6,
    Math.min(100 - left, ((range.end - range.start) / duration) * 100),
  );
  return (
    <span
      className="timeline-marker"
      style={{ left: `${left}%`, width: `${width}%` }}
      title={`${formatClock(range.start)}–${formatClock(range.end)}：${range.reason}`}
    />
  );
}

function formatClock(value: number): string {
  if (!Number.isFinite(value)) return "00:00";
  const minutes = Math.floor(value / 60);
  const seconds = Math.floor(value % 60);
  return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
}

