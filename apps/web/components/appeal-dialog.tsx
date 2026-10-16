"use client";

import { CircleAlert, Send, X } from "lucide-react";
import { FormEvent, useState } from "react";

import { ApiError, submitAppeal } from "@/lib/api";

export function AppealDialog({ jobId }: { jobId: string }) {
  const [open, setOpen] = useState(false);
  const [reason, setReason] = useState("");
  const [contact, setContact] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      const result = await submitAppeal(jobId, {
        reason,
        contact: contact.trim() || undefined,
      });
      setSuccess(`申诉已受理：${result.appeal_id}`);
      setSubmitting(false);
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : "申诉提交失败。");
      setSubmitting(false);
    }
  }

  return (
    <>
      <button className="button button-secondary" onClick={() => setOpen(true)} type="button">
        提交申诉
      </button>
      {open ? (
        <div className="modal-backdrop" role="presentation" onMouseDown={() => setOpen(false)}>
          <div
            className="modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="appeal-title"
            onMouseDown={(event) => event.stopPropagation()}
          >
            <div className="modal-header">
              <h2 className="panel-title" id="appeal-title">
                检测结果申诉
              </h2>
              <button
                className="icon-button"
                type="button"
                onClick={() => setOpen(false)}
                aria-label="关闭"
              >
                <X size={16} />
              </button>
            </div>
            <form onSubmit={handleSubmit}>
              <div className="modal-body">
                {success ? (
                  <div className="alert" style={{ marginBottom: 16 }}>
                    {success}
                  </div>
                ) : null}
                {error ? (
                  <div className="alert alert-error" style={{ marginBottom: 16 }}>
                    <CircleAlert size={15} style={{ verticalAlign: "-2px", marginRight: 6 }} />
                    {error}
                  </div>
                ) : null}
                <div className="stack" style={{ gap: 15 }}>
                  <div className="field">
                    <label htmlFor="appeal-reason">申诉理由</label>
                    <textarea
                      id="appeal-reason"
                      className="textarea"
                      minLength={10}
                      required
                      value={reason}
                      onChange={(event) => setReason(event.target.value)}
                      placeholder="说明素材来源、拍摄过程或对具体证据的异议。"
                    />
                  </div>
                  <div className="field">
                    <label htmlFor="appeal-contact">联系方式（可选）</label>
                    <input
                      id="appeal-contact"
                      className="input"
                      type="text"
                      maxLength={320}
                      value={contact}
                      onChange={(event) => setContact(event.target.value)}
                    />
                  </div>
                </div>
              </div>
              <div className="modal-footer">
                <button className="button button-secondary" type="button" onClick={() => setOpen(false)}>
                  取消
                </button>
                <button
                  className="button button-primary"
                  type="submit"
                  disabled={submitting || reason.trim().length < 10}
                >
                  <Send size={15} />
                  {submitting ? "提交中…" : "提交申诉"}
                </button>
              </div>
            </form>
          </div>
        </div>
      ) : null}
    </>
  );
}

