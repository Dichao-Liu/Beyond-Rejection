from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models


# Risk-semantic definitions and text anchors

RISK_SLOTS = (
    "manual_distraction",
    "visual_distraction",
    "control_disruption",
    "object_interaction",
    "task_irrelevant_engagement",
)

TEXT_ANCHORS: Dict[str, Tuple[str, str]] = {
    "manual_distraction": (
        "hands engaged with a non-driving object",
        "hands not fully available for driving",
    ),
    "visual_distraction": (
        "driver looking away from the forward roadway",
        "visual attention deviated from driving",
    ),
    "control_disruption": (
        "reduced steering control",
        "driver not maintaining stable control of the vehicle",
    ),
    "object_interaction": (
        "interaction with a non-driving object",
        "driver holding or using an object unrelated to driving",
    ),
    "task_irrelevant_engagement": (
        "driver engaged in a task unrelated to driving",
        "driver performing a non-driving activity",
    ),
}

CLASS_RISK_PRIORS: Dict[str, Tuple[float, float, float, float, float]] = {
    "safe_driving": (0.1, 0.1, 0.1, 0.1, 0.1),
    "texting_right": (0.9, 0.5, 0.9, 0.9, 0.9),
    "talking_on_the_phone_right": (0.9, 0.5, 0.5, 0.9, 0.9),
    "texting_left": (0.9, 0.5, 0.9, 0.9, 0.9),
    "talking_on_the_phone_left": (0.9, 0.5, 0.5, 0.9, 0.9),
    "operating_the_radio": (0.5, 0.5, 0.9, 0.5, 0.5),
    "drinking": (0.5, 0.1, 0.5, 0.9, 0.9),
    "reaching_behind": (0.5, 0.9, 0.9, 0.5, 0.9),
    "hair_and_makeup": (0.9, 0.9, 0.9, 0.9, 0.9),
    "talking_to_passenger": (0.1, 0.9, 0.5, 0.1, 0.5),
}

VISION_LANGUAGE_ALPHA = 0.70


def minmax_normalize_semantics(
    scores: torch.Tensor,
    minimums: torch.Tensor,
    maximums: torch.Tensor,
) -> torch.Tensor:
    denominator = (maximums - minimums).clamp_min(1e-6)
    return ((scores - minimums) / denominator).clamp(0.0, 1.0)


def fuse_class_risk_priors(
    semantic_scores: torch.Tensor,
    class_names: Sequence[str],
    alpha: float = VISION_LANGUAGE_ALPHA,
    class_priors: Mapping[str, Sequence[float]] = CLASS_RISK_PRIORS,
) -> torch.Tensor:
    if semantic_scores.ndim != 2 or semantic_scores.shape[1] != len(RISK_SLOTS):
        raise ValueError("semantic_scores must have shape [batch, 5].")
    if len(class_names) != semantic_scores.shape[0]:
        raise ValueError("class_names must match the batch size.")
    priors = []
    for class_name in class_names:
        if class_name not in class_priors:
            raise KeyError(f"Missing risk prior for class: {class_name}")
        values = tuple(float(value) for value in class_priors[class_name])
        if len(values) != len(RISK_SLOTS):
            raise ValueError(f"Risk prior for {class_name} must contain five values.")
        priors.append(values)
    prior_tensor = semantic_scores.new_tensor(priors)
    return alpha * semantic_scores + (1.0 - alpha) * prior_tensor


# MobileNetV2 fast branch

class FastRiskSemanticBranch(nn.Module):
    def __init__(
        self,
        num_known_classes: int = 6,
        num_risk_slots: int = 5,
        dropout: float = 0.2,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        try:
            weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.mobilenet_v2(weights=weights)
        except (AttributeError, TypeError):
            backbone = models.mobilenet_v2(pretrained=pretrained)
        feature_dim = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        self.action_head = nn.Linear(feature_dim, num_known_classes)
        self.semantic_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_risk_slots),
        )

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.backbone(images)
        action_logits = self.action_head(features)
        semantic_logits = self.semantic_head(features)
        return action_logits, semantic_logits, features


# Uncertainty-score computation

def compute_uncertainty_scores(action_logits: torch.Tensor) -> torch.Tensor:
    if action_logits.ndim != 2 or action_logits.shape[1] < 2:
        raise ValueError("action_logits must have shape [batch, classes] with at least two classes.")
    probabilities = torch.softmax(action_logits, dim=1)
    top1_probability = probabilities.max(dim=1).values
    top2_logits = torch.topk(action_logits, k=2, dim=1).values
    logit_margin = top2_logits[:, 0] - top2_logits[:, 1]
    energy = -torch.logsumexp(action_logits, dim=1)
    return torch.stack((top1_probability, logit_margin, energy), dim=1)


# Risk-region construction

FACE_HEAD_IDS = (1, 2, 9, 11)
ARM_HAND_IDS = (12, 13)
TORSO_CONTROL_IDS = (3, 4, 16, 12, 13)


def _clamp_int(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def _mask_from_ids(segmentation: np.ndarray, class_ids: Sequence[int]) -> np.ndarray:
    mask = np.zeros_like(segmentation, dtype=bool)
    for class_id in class_ids:
        mask |= segmentation == int(class_id)
    return mask.astype(np.uint8)


def _bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _expand_bbox(
    bbox: Tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    scale_x: float,
    scale_y: float,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    center_x = 0.5 * (x1 + x2)
    center_y = 0.5 * (y1 + y2)
    width = max(x2 - x1, 16)
    height = max(y2 - y1, 16)
    new_width = max(int(round(width * scale_x)), 16)
    new_height = max(int(round(height * scale_y)), 16)
    nx1 = _clamp_int(int(round(center_x - new_width / 2)), 0, image_width - 1)
    ny1 = _clamp_int(int(round(center_y - new_height / 2)), 0, image_height - 1)
    nx2 = _clamp_int(int(round(center_x + new_width / 2)), nx1 + 1, image_width)
    ny2 = _clamp_int(int(round(center_y + new_height / 2)), ny1 + 1, image_height)
    return nx1, ny1, nx2, ny2


def _fallback_face_bbox(width: int, height: int) -> Tuple[int, int, int, int]:
    box_width = int(round(width * 0.28))
    box_height = int(round(height * 0.28))
    x1 = _clamp_int(int(round(width * 0.5 - box_width / 2)), 0, width - 1)
    y1 = _clamp_int(int(round(height * 0.08)), 0, height - 1)
    x2 = _clamp_int(x1 + box_width, x1 + 1, width)
    y2 = _clamp_int(y1 + box_height, y1 + 1, height)
    return x1, y1, x2, y2


def _fallback_arm_bbox(width: int, height: int) -> Tuple[int, int, int, int]:
    box_width = int(round(width * 0.45))
    box_height = int(round(height * 0.30))
    x1 = _clamp_int(int(round(width * 0.5 - box_width / 2)), 0, width - 1)
    y1 = _clamp_int(int(round(height * 0.32)), 0, height - 1)
    x2 = _clamp_int(x1 + box_width, x1 + 1, width)
    y2 = _clamp_int(y1 + box_height, y1 + 1, height)
    return x1, y1, x2, y2


def _control_bbox(
    segmentation: Optional[np.ndarray],
    width: int,
    height: int,
) -> Tuple[int, int, int, int]:
    torso = None
    if segmentation is not None:
        torso = _bbox_from_mask(_mask_from_ids(segmentation, TORSO_CONTROL_IDS))
    if torso is not None:
        x1, y1, x2, y2 = torso
        center_x = 0.5 * (x1 + x2)
        center_y = 0.5 * (y1 + y2)
        box_width = int(round(max((x2 - x1) * 1.2, width * 0.28)))
        box_height = int(round(max((y2 - y1) * 0.9, height * 0.20)))
        center_y = min(height - 1, center_y + 0.25 * (y2 - y1))
    else:
        box_width = int(round(width * 0.35))
        box_height = int(round(height * 0.22))
        center_x = width * 0.5
        center_y = height * 0.62
    x1 = _clamp_int(int(round(center_x - box_width / 2)), 0, width - 1)
    y1 = _clamp_int(int(round(center_y - box_height / 2)), 0, height - 1)
    x2 = _clamp_int(int(round(center_x + box_width / 2)), x1 + 1, width)
    y2 = _clamp_int(int(round(center_y + box_height / 2)), y1 + 1, height)
    return x1, y1, x2, y2


def construct_risk_regions(
    image: Image.Image,
    segmentation: Optional[np.ndarray],
) -> Dict[str, object]:
    width, height = image.size
    if segmentation is None:
        face_head_bbox = _fallback_face_bbox(width, height)
        arm_hand_bbox = _fallback_arm_bbox(width, height)
    else:
        face_head_bbox = _bbox_from_mask(_mask_from_ids(segmentation, FACE_HEAD_IDS))
        arm_hand_bbox = _bbox_from_mask(_mask_from_ids(segmentation, ARM_HAND_IDS))
        if face_head_bbox is None:
            face_head_bbox = _fallback_face_bbox(width, height)
        else:
            face_head_bbox = _expand_bbox(face_head_bbox, width, height, 1.35, 1.35)
        if arm_hand_bbox is None:
            arm_hand_bbox = _fallback_arm_bbox(width, height)
        else:
            arm_hand_bbox = _expand_bbox(arm_hand_bbox, width, height, 1.45, 1.35)
    control_bbox = _control_bbox(segmentation, width, height)
    return {
        "face_head_bbox": face_head_bbox,
        "arm_hand_bbox": arm_hand_bbox,
        "control_bbox": control_bbox,
        "face_head_crop": image.crop(face_head_bbox),
        "arm_hand_crop": image.crop(arm_hand_bbox),
        "control_crop": image.crop(control_bbox),
    }


# Local semantic-support computation

class LocalRiskSemanticScorer:
    def __init__(
        self,
        device: Optional[torch.device] = None,
        model_name: str = "ViT-B-16",
        pretrained: str = "laion2b_s34b_b88k",
        slots: Sequence[str] = RISK_SLOTS,
        text_anchors: Mapping[str, Sequence[str]] = TEXT_ANCHORS,
    ) -> None:
        try:
            import open_clip
        except ImportError as exc:
            raise ImportError("open_clip_torch is required for semantic-support computation.") from exc
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.slots = tuple(slots)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=str(self.device),
        )
        self.model.eval()
        tokenizer = open_clip.get_tokenizer(model_name)
        prompts = []
        prompt_slots = []
        for slot in self.slots:
            if slot not in text_anchors or len(text_anchors[slot]) == 0:
                raise ValueError(f"Missing text anchors for risk slot: {slot}")
            for prompt in text_anchors[slot]:
                prompts.append(str(prompt))
                prompt_slots.append(slot)
        self.prompt_slots = tuple(prompt_slots)
        with torch.no_grad():
            tokens = tokenizer(prompts).to(self.device)
            text_features = self.model.encode_text(tokens)
            self.text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def score_crops(self, crops: Sequence[Image.Image]) -> torch.Tensor:
        if len(crops) == 0:
            return torch.empty((0, len(self.slots)), dtype=torch.float32)
        batch = torch.stack([self.preprocess(crop.convert("RGB")) for crop in crops], dim=0)
        image_features = self.model.encode_image(batch.to(self.device))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        similarities = image_features @ self.text_features.T
        outputs = []
        for slot in self.slots:
            indices = [index for index, value in enumerate(self.prompt_slots) if value == slot]
            outputs.append(similarities[:, indices].mean(dim=1))
        return torch.stack(outputs, dim=1)

    @torch.no_grad()
    def score_regions(
        self,
        image: Image.Image,
        segmentation: Optional[np.ndarray],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        regions = construct_risk_regions(image, segmentation)
        scores = self.score_crops(
            (
                regions["face_head_crop"],
                regions["arm_hand_crop"],
                regions["control_crop"],
            )
        )
        return scores[0:1], scores[1:2], scores[2:3]


# 23-dimensional evidence composition

def compose_slow_evidence(
    uncertainty_scores: torch.Tensor,
    global_semantics: torch.Tensor,
    face_head_semantics: torch.Tensor,
    arm_hand_semantics: torch.Tensor,
    control_semantics: torch.Tensor,
) -> torch.Tensor:
    tensors = (
        uncertainty_scores,
        global_semantics,
        face_head_semantics,
        arm_hand_semantics,
        control_semantics,
    )
    if any(tensor.ndim != 2 for tensor in tensors):
        raise ValueError("All evidence tensors must have shape [batch, features].")
    batch_sizes = {tensor.shape[0] for tensor in tensors}
    if len(batch_sizes) != 1:
        raise ValueError("All evidence tensors must have the same batch size.")
    expected_dimensions = (3, 5, 5, 5, 5)
    actual_dimensions = tuple(tensor.shape[1] for tensor in tensors)
    if actual_dimensions != expected_dimensions:
        raise ValueError(f"Expected feature dimensions {expected_dimensions}, received {actual_dimensions}.")
    evidence = torch.cat(tensors, dim=1)
    if evidence.shape[1] != 23:
        raise RuntimeError("The slow-verifier evidence must contain 23 dimensions.")
    return evidence


# Three-way slow verifier

STATE_NAMES = ("known", "semantic_unknown", "pseudo_unknown")


class ThreeWaySlowVerifier(nn.Module):
    def __init__(
        self,
        input_dim: int = 23,
        hidden_dim_1: int = 64,
        hidden_dim_2: int = 32,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if input_dim != 23:
            raise ValueError("The three-way slow verifier expects 23 input features.")
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim_2, len(STATE_NAMES)),
        )

    def forward(self, evidence: torch.Tensor) -> torch.Tensor:
        if evidence.ndim != 2 or evidence.shape[1] != 23:
            raise ValueError("evidence must have shape [batch, 23].")
        return self.network(evidence)
