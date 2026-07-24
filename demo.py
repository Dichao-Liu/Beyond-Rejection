from __future__ import annotations

from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from method import (
    FastRiskSemanticBranch,
    LocalRiskSemanticScorer,
    ThreeWaySlowVerifier,
    compose_slow_evidence,
    compute_uncertainty_scores,
    construct_risk_regions,
)


DEFAULT_IMAGE = Path("aucd/test/drinking/1007.jpg")
PARSER_MODEL = "fashn-ai/fashn-human-parser"


def resolve_image_path() -> Path:
    if len(sys.argv) > 1:
        path = Path(sys.argv[1]).expanduser()
        if path.is_file():
            return path
        raise FileNotFoundError(f"Image not found: {path}")
    candidates = (
        Path.cwd() / DEFAULT_IMAGE,
        Path(__file__).resolve().parent / DEFAULT_IMAGE,
        Path(__file__).resolve().parent.parent / DEFAULT_IMAGE,
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"Default image not found: {DEFAULT_IMAGE}. Run: python demo.py /path/to/image.jpg"
    )


def predict_segmentation(image: Image.Image, device: torch.device):
    try:
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
    except ImportError as exc:
        raise ImportError("transformers is required for the demo.") from exc
    processor = SegformerImageProcessor.from_pretrained(PARSER_MODEL)
    model = SegformerForSemanticSegmentation.from_pretrained(PARSER_MODEL).to(device).eval()
    inputs = processor(images=image, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
        logits = F.interpolate(logits, size=image.size[::-1], mode="bilinear", align_corners=False)
    return logits.argmax(dim=1).squeeze(0).cpu().numpy()


def main() -> None:
    image_path = resolve_image_path()
    image = Image.open(image_path).convert("RGB")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose(
        (
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        )
    )
    images = transform(image).unsqueeze(0).to(device)

    fast = FastRiskSemanticBranch(pretrained=False).to(device).eval()
    slow = ThreeWaySlowVerifier().to(device).eval()
    with torch.no_grad():
        action_logits, semantic_logits, features = fast(images)
        uncertainty = compute_uncertainty_scores(action_logits)
        global_semantics = torch.sigmoid(semantic_logits)

    segmentation = predict_segmentation(image, device)
    regions = construct_risk_regions(image, segmentation)
    scorer = LocalRiskSemanticScorer(device=device)
    face_head, arm_hand, control = scorer.score_regions(image, segmentation)
    evidence = compose_slow_evidence(
        uncertainty,
        global_semantics,
        face_head,
        arm_hand,
        control,
    )
    with torch.no_grad():
        state_logits = slow(evidence)

    print(f"image: {image_path}")
    print(f"action_logits: {tuple(action_logits.shape)}")
    print(f"semantic_logits: {tuple(semantic_logits.shape)}")
    print(f"features: {tuple(features.shape)}")
    print(f"uncertainty: {tuple(uncertainty.shape)}")
    print(f"face_head_bbox: {regions['face_head_bbox']}")
    print(f"arm_hand_bbox: {regions['arm_hand_bbox']}")
    print(f"control_bbox: {regions['control_bbox']}")
    print(f"local_semantics: {tuple(face_head.shape)}, {tuple(arm_hand.shape)}, {tuple(control.shape)}")
    print(f"evidence: {tuple(evidence.shape)}")
    print(f"state_logits: {tuple(state_logits.shape)}")
    print("status: ok")


if __name__ == "__main__":
    main()
