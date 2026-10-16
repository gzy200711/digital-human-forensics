from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from apps.api.services.media import MediaProbe, sample_video_frames


@dataclass
class FaceSample:
    timestamp: float
    frame_index: int
    frame: np.ndarray
    bbox: tuple[int, int, int, int]
    crop: np.ndarray
    blur_score: float
    brightness_score: float
    pose_score: float
    occlusion_score: float
    sharpness_raw: float
    skin_ratio: float

    @property
    def face_crop(self) -> np.ndarray:
        """严格人脸框内的像素，不含上下文边距，供时序/纹理分析使用。"""
        frame_height, frame_width = self.frame.shape[:2]
        x, y, width, height = self.bbox
        x1 = max(0, min(int(x), frame_width))
        y1 = max(0, min(int(y), frame_height))
        x2 = max(x1, min(int(x + width), frame_width))
        y2 = max(y1, min(int(y + height), frame_height))
        return self.frame[y1:y2, x1:x2]


@dataclass
class FaceQualityResult:
    score: float
    status: str
    face_detected_ratio: float
    factors: list[dict] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    samples: list[FaceSample] = field(default_factory=list)
    sampled_frames: int = 0


class FaceQualityAnalyzer:
    def __init__(
        self,
        *,
        pass_threshold: float,
        marginal_threshold: float,
    ) -> None:
        self.pass_threshold = pass_threshold
        self.marginal_threshold = marginal_threshold
        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.eye_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
        )

    def analyze(
        self,
        probe: MediaProbe,
        *,
        sample_fps: float,
        max_frames: int,
    ) -> FaceQualityResult:
        if probe.video_stream is None:
            return self._empty_result(
                score=0.0,
                reason="输入不含视频轨，当前质量门控无法评估人脸可鉴定性。",
            )

        frames = sample_video_frames(
            probe.path,
            sample_fps=sample_fps,
            max_frames=max_frames,
            duration_seconds=probe.duration_seconds,
        )
        samples: list[FaceSample] = []
        for frame in frames:
            sample = self._detect_largest_face(frame.timestamp, frame.index, frame.image)
            if sample is not None:
                samples.append(sample)

        detected_ratio = len(samples) / len(frames) if frames else 0.0
        if not samples:
            return self._empty_result(
                score=0.0,
                reason="未检测到可跟踪的人脸，无法作出视觉判断。",
                detected_ratio=0.0,
                sampled_frames=len(frames),
            )

        face_area_ratios = []
        blur_scores = []
        brightness_scores = []
        pose_scores = []
        occlusion_scores = []
        for sample in samples:
            _, _, width, height = sample.bbox
            frame_height, frame_width = sample.frame.shape[:2]
            face_area_ratios.append((width * height) / max(1, frame_width * frame_height))
            blur_scores.append(sample.blur_score)
            brightness_scores.append(sample.brightness_score)
            pose_scores.append(sample.pose_score)
            occlusion_scores.append(sample.occlusion_score)

        median_face_ratio = float(np.median(face_area_ratios))
        face_size_score = _face_size_score(median_face_ratio)
        blur_score = float(np.median(blur_scores))
        brightness_score = float(np.median(brightness_scores))
        pose_score = float(np.median(pose_scores))
        occlusion_score = float(np.median(occlusion_scores))
        detection_score = min(1.0, detected_ratio / 0.72)

        score = float(
            np.clip(
                0.20 * detection_score
                + 0.22 * face_size_score
                + 0.22 * blur_score
                + 0.12 * brightness_score
                + 0.12 * pose_score
                + 0.12 * occlusion_score,
                0,
                1,
            )
        )
        if score >= self.pass_threshold and detected_ratio >= 0.45:
            status = "pass"
        elif score >= self.marginal_threshold and detected_ratio >= 0.2:
            status = "marginal"
        else:
            status = "fail"

        factors = [
            self._factor(
                "face_detection",
                round(detected_ratio, 4),
                _status(detected_ratio, 0.45, 0.2),
                f"抽帧中 {detected_ratio:.0%} 的画面检测到人脸。",
            ),
            self._factor(
                "face_size",
                round(median_face_ratio, 4),
                _status(median_face_ratio, 0.05, 0.018),
                f"人脸面积中位数占画面 {median_face_ratio:.1%}。",
            ),
            self._factor(
                "sharpness",
                round(blur_score, 4),
                _status(blur_score, 0.62, 0.38),
                "清晰度综合拉普拉斯方差归一化结果。",
            ),
            self._factor(
                "brightness",
                round(brightness_score, 4),
                _status(brightness_score, 0.6, 0.35),
                "人脸区域曝光是否适合纹理分析。",
            ),
            self._factor(
                "pose",
                round(pose_score, 4),
                _status(pose_score, 0.68, 0.45),
                "基于正面眼部可检测性估计的姿态可用性。",
            ),
            self._factor(
                "occlusion",
                round(occlusion_score, 4),
                _status(occlusion_score, 0.6, 0.35),
                "基于肤色可见比例的遮挡近似估计，不等于语义遮挡检测。",
            ),
        ]
        reasons = _quality_reasons(
            detected_ratio=detected_ratio,
            face_ratio=median_face_ratio,
            blur_score=blur_score,
            brightness_score=brightness_score,
            pose_score=pose_score,
            occlusion_score=occlusion_score,
        )
        return FaceQualityResult(
            score=score,
            status=status,
            face_detected_ratio=detected_ratio,
            factors=factors,
            reasons=reasons,
            samples=samples,
            sampled_frames=len(frames),
        )

    def _detect_largest_face(
        self,
        timestamp: float,
        frame_index: int,
        frame: np.ndarray,
    ) -> FaceSample | None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.12,
            minNeighbors=5,
            minSize=(36, 36),
        )
        if len(faces) == 0:
            return None
        x, y, width, height = max(faces, key=lambda item: int(item[2]) * int(item[3]))
        frame_height, frame_width = gray.shape
        margin_x = int(width * 0.16)
        margin_y = int(height * 0.20)
        x1 = max(0, int(x) - margin_x)
        y1 = max(0, int(y) - margin_y)
        x2 = min(frame_width, int(x + width) + margin_x)
        y2 = min(frame_height, int(y + height) + margin_y)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        sharpness_raw = float(cv2.Laplacian(crop_gray, cv2.CV_64F).var())
        blur_score = float(1 - math.exp(-sharpness_raw / 140.0))
        mean_brightness = float(np.mean(crop_gray))
        brightness_score = _brightness_score(mean_brightness)
        pose_score = self._pose_score(crop_gray)
        skin_ratio = _skin_ratio(crop)
        occlusion_score = float(np.clip((skin_ratio - 0.16) / 0.36, 0, 1))
        return FaceSample(
            timestamp=timestamp,
            frame_index=frame_index,
            frame=frame,
            bbox=(x1, y1, x2 - x1, y2 - y1),
            crop=crop,
            blur_score=blur_score,
            brightness_score=brightness_score,
            pose_score=pose_score,
            occlusion_score=occlusion_score,
            sharpness_raw=sharpness_raw,
            skin_ratio=skin_ratio,
        )

    def _pose_score(self, face_gray: np.ndarray) -> float:
        height, width = face_gray.shape
        upper = face_gray[: int(height * 0.65), :]
        eyes = self.eye_detector.detectMultiScale(
            upper,
            scaleFactor=1.12,
            minNeighbors=4,
            minSize=(max(8, width // 12), max(8, height // 14)),
        )
        if len(eyes) >= 2:
            return 1.0
        if len(eyes) == 1:
            return 0.68
        return 0.43

    @staticmethod
    def _factor(name: str, value: float, status: str, explanation: str) -> dict:
        return {
            "name": name,
            "value": value,
            "status": status,
            "explanation": explanation,
        }

    def _empty_result(
        self,
        *,
        score: float,
        reason: str,
        detected_ratio: float = 0.0,
        sampled_frames: int = 0,
    ) -> FaceQualityResult:
        return FaceQualityResult(
            score=score,
            status="fail",
            face_detected_ratio=detected_ratio,
            factors=[],
            reasons=[reason],
            samples=[],
            sampled_frames=sampled_frames,
        )


def _brightness_score(mean_brightness: float) -> float:
    if mean_brightness < 35:
        return float(np.clip(mean_brightness / 70, 0, 1))
    if mean_brightness > 230:
        return float(np.clip((255 - mean_brightness) / 45, 0, 1))
    return 1.0


def _face_size_score(face_area_ratio: float) -> float:
    # 面积比 5% 左右达到高可用区，低于 1.8% 基本不可鉴定。
    return float(np.clip((face_area_ratio - 0.012) / 0.055, 0, 1))


def _skin_ratio(crop: np.ndarray) -> float:
    ycrcb = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)
    mask = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 173, 127]))
    return float(np.count_nonzero(mask) / max(1, mask.size))


def _status(value: float, good: float, bad: float) -> str:
    if value >= good:
        return "good"
    if value >= bad:
        return "warning"
    return "bad"


def _quality_reasons(
    *,
    detected_ratio: float,
    face_ratio: float,
    blur_score: float,
    brightness_score: float,
    pose_score: float,
    occlusion_score: float,
) -> list[str]:
    reasons: list[str] = []
    if detected_ratio < 0.45:
        reasons.append(f"只有 {detected_ratio:.0%} 的抽帧检测到人脸，目标可能被遮挡或频繁出画。")
    if face_ratio < 0.018:
        reasons.append(f"人脸面积中位数仅 {face_ratio:.2%}，细节信号不可靠。")
    if blur_score < 0.5:
        reasons.append("人脸区域清晰度偏低，压缩或运动模糊会污染伪影判断。")
    if brightness_score < 0.55:
        reasons.append("人脸区域曝光不理想，纹理与肤色信号可信度下降。")
    if pose_score < 0.55:
        reasons.append("样本以侧脸或眼部遮挡为主，姿态一致性判断受限。")
    if occlusion_score < 0.45:
        reasons.append("人脸区域可见比例不足，可能存在口罩、手部或其他遮挡。")
    return reasons
