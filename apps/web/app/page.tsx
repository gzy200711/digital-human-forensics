import {
  Activity,
  AudioLines,
  CircleGauge,
  FileCheck2,
  Fingerprint,
  ScanFace,
  Scale,
  Waves,
} from "lucide-react";

import { AppHeader } from "@/components/app-header";
import { SubmitForm } from "@/components/submit-form";

const VERDICTS = [
  {
    key: "trusted",
    label: "可信",
    tone: "badge-trust",
    definition:
      "在本次检测覆盖的模态、时段和质量范围内，未发现足够强的生成合成证据。",
    boundary: "这不是对原始真实性的证明，也不代表素材未经编辑。",
  },
  {
    key: "uncertain",
    label: "无法确定",
    tone: "badge-uncertain",
    definition:
      "证据不足、信号互相矛盾、样本超出已验证分布，或可鉴定性不足。",
    boundary: "平台不下方向性结论，报告会列出具体的拒判原因。",
  },
  {
    key: "suspected",
    label: "疑似生成合成",
    tone: "badge-suspect",
    definition:
      "多个独立信号在同一时段或同一模态指向生成合成，且越过已锁定阈值。",
    boundary: "不写成“此人一定是 AI”，也不判断身份、意图或法律责任。",
  },
] as const;

export default function HomePage() {
  return (
    <div className="app-shell">
      <AppHeader />
      <main className="page">
        <div className="page-header">
          <div>
            <div className="page-eyebrow">Evidence-first Forensic Analysis</div>
            <h1 className="page-title">数字人 / 生成合成内容鉴别</h1>
            <p className="page-subtitle">
              按模态分别输出画面、音频、跨模态与溯源结论，再综合为三级判定；
              每条证据都带 ID、时间范围、区域、模型版本与不确定度。
            </p>
          </div>
          <span className="badge badge-brand">
            <Activity size={13} />
            异步检测
          </span>
        </div>

        <div className="submit-grid">
          <SubmitForm />
          <aside className="panel">
            <div className="panel-header">
              <h2 className="panel-title">检测口径</h2>
              <CircleGauge size={18} color="#667085" />
            </div>
            <div className="panel-body">
              <div className="ops-list">
                <div className="ops-row">
                  <span className="ops-label">L0 溯源</span>
                  <span className="ops-value">ffprobe + C2PA，仅作阳性证据</span>
                </div>
                <div className="ops-row">
                  <span className="ops-label">L1 质量门控</span>
                  <span className="ops-value">只决定分支可用性，不代替判定</span>
                </div>
                <div className="ops-row">
                  <span className="ops-label">L2 画面</span>
                  <span className="ops-value">帧级伪影 + 人脸区域质量</span>
                </div>
                <div className="ops-row">
                  <span className="ops-label">L3 时序</span>
                  <span className="ops-value">人脸框内一致性与微动</span>
                </div>
                <div className="ops-row">
                  <span className="ops-label">L5 音频 / 跨模态</span>
                  <span className="ops-value">反欺骗与唇音同步分开评测</span>
                </div>
                <div className="ops-row">
                  <span className="ops-label">L7 融合</span>
                  <span className="ops-value">同模态判冲突，跨模态分别成像</span>
                </div>
              </div>

              <div className="alert" style={{ marginTop: 18 }}>
                元数据、C2PA 与水印属于溯源信号：有效的生成声明可作为阳性证据，
                缺失或被清除不能作为“真实”证据。
              </div>

              <div className="coverage-chips">
                <span className="coverage-chip">
                  <ScanFace size={14} />
                  画面（人脸区域）
                </span>
                <span className="coverage-chip">
                  <AudioLines size={14} />
                  音频反欺骗
                </span>
                <span className="coverage-chip">
                  <Waves size={14} />
                  唇音跨模态
                </span>
                <span className="coverage-chip">
                  <Fingerprint size={14} />
                  溯源与元数据
                </span>
              </div>

              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "1fr 1fr",
                  gap: 10,
                  marginTop: 16,
                }}
              >
                <div className="metric">
                  <div className="metric-label">最大时长</div>
                  <div className="metric-value">15 min</div>
                </div>
                <div className="metric">
                  <div className="metric-label">原始留存</div>
                  <div className="metric-value">1–30 d</div>
                </div>
                <div className="metric">
                  <div className="metric-label">输入类型</div>
                  <div className="metric-value">A/V</div>
                </div>
                <div className="metric">
                  <div className="metric-label">审计日志</div>
                  <div className="metric-value">全量</div>
                </div>
              </div>

              <div className="panel-footnote">
                <FileCheck2 size={15} />
                结构化报告含覆盖率、拒判原因、模型快照与阈值策略。
              </div>
            </div>
          </aside>
        </div>

        <section className="panel" style={{ marginTop: 18 }}>
          <div className="panel-header">
            <h2 className="panel-title">三级结论的含义</h2>
            <span className="badge badge-neutral">
              <Scale size={13} />
              带范围限定的判定
            </span>
          </div>
          <div className="panel-body">
            <div className="verdict-defs">
              {VERDICTS.map((item) => (
                <article className="verdict-def" key={item.key} data-tone={item.key}>
                  <span className={`badge ${item.tone}`}>{item.label}</span>
                  <p className="verdict-def-text">{item.definition}</p>
                  <p className="verdict-def-boundary">{item.boundary}</p>
                </article>
              ))}
            </div>
            <ul className="plain-list platform-boundaries">
              <li>
                平台不判断人物身份、创作意图、道德属性或法律责任，也不把“未检出异常”表述为“已证明真实”。
              </li>
              <li>
                人脸分支不可用只关闭该分支；音频、跨模态或溯源分支仍可用时，不会自动把整条媒体判为“无法确定”。
              </li>
              <li>无音轨只关闭音频与唇音分支，并在报告中显式说明覆盖范围下降。</li>
              <li>
                FPR 指标按统计门槛报告；真实阴性样本不足时会写“证据不足”，而不是只给点估计。
              </li>
            </ul>
          </div>
        </section>
      </main>
    </div>
  );
}
