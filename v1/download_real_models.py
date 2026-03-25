#!/usr/bin/env python3
"""
download_real_models.py -- 실제 TFLite 모델 다운로더
urllib (표준 라이브러리)만 사용하여 TF Hub / TFLite 공개 모델을 다운로드한다.
다운로드된 모델은 tflite_sast.py로 분석하여 실제 FPR을 측정할 수 있다.

사용법:
    python download_real_models.py               # 기본 디렉터리(real_models/)
    python download_real_models.py --dir mydir   # 지정 디렉터리
    python download_real_models.py --analyze     # 다운로드 후 즉시 분석
"""

import urllib.request
import urllib.error
import os
import time
import json
import argparse
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# 공개 TFLite 모델 목록
# 출처: TFLite 공식 예제, TF Hub, TFLite 모델 zoo
# 모두 비인증 공개 URL이며 별도 pip 없이 다운로드 가능
# ─────────────────────────────────────────────────────────────
REAL_MODELS = [
    # ── 이미지 분류 계열 (task_library 경로 — 공개 접근 가능) ──
    {
        "id": "R_B01",
        "name": "MobileNetV1 1.0_224 quant (ImageNet)",
        "url": "https://storage.googleapis.com/download.tensorflow.org/"
               "models/tflite/task_library/image_classification/android/"
               "mobilenet_v1_1.0_224_quantized_1_metadata_1.tflite",
        "expected_vuln": False,
        "note": "양자화 MobileNetV1 분류 모델",
    },
    {
        "id": "R_B02",
        "name": "EfficientNet Lite0 int8 (ImageNet)",
        "url": "https://storage.googleapis.com/download.tensorflow.org/"
               "models/tflite/task_library/image_classification/android/"
               "efficientnet_lite0_int8_2.tflite",
        "expected_vuln": False,
        "note": "EfficientNet Lite0 int8 분류 모델",
    },
    {
        "id": "R_B03",
        "name": "EfficientDet-Lite0 (Object Detection)",
        "url": "https://storage.googleapis.com/download.tensorflow.org/"
               "models/tflite/task_library/object_detection/android/"
               "lite-model_efficientdet_lite0_detection_metadata_1.tflite",
        "expected_vuln": False,
        "note": "다중 출력 탐지 모델",
    },
    {
        "id": "R_B04",
        "name": "DeepLab v3 (Segmentation)",
        "url": "https://storage.googleapis.com/download.tensorflow.org/"
               "models/tflite/gpu/deeplabv3_257_mv_gpu.tflite",
        "expected_vuln": False,
        "note": "세그멘테이션 모델 — is_variable 텐서 존재 가능",
    },
    {
        "id": "R_B05",
        "name": "EfficientNet Lite1 int8 (ImageNet)",
        "url": "https://storage.googleapis.com/download.tensorflow.org/"
               "models/tflite/task_library/image_classification/android/"
               "efficientnet_lite1_int8_2.tflite",
        "expected_vuln": False,
        "note": "EfficientNet Lite1 int8 분류 모델",
    },
    {
        "id": "R_B06",
        "name": "EfficientNet Lite2 int8 (ImageNet)",
        "url": "https://storage.googleapis.com/download.tensorflow.org/"
               "models/tflite/task_library/image_classification/android/"
               "efficientnet_lite2_int8_2.tflite",
        "expected_vuln": False,
        "note": "EfficientNet Lite2 int8 분류 모델",
    },
    {
        "id": "R_B07",
        "name": "MobileNetV1 1.0_224 float32 GPU (ImageNet)",
        "url": "https://storage.googleapis.com/download.tensorflow.org/"
               "models/tflite/gpu/mobilenet_v1_1.0_224.tflite",
        "expected_vuln": False,
        "note": "float32 MobileNetV1 GPU 최적화 모델",
    },
    {
        "id": "R_B08",
        "name": "MobileNetV1 SSD (COCO Object Detection)",
        "url": "https://storage.googleapis.com/download.tensorflow.org/"
               "models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip",
        "expected_vuln": False,
        "note": "양자화 탐지 모델 (.zip 내 .tflite 포함)",
        "is_zip": True,
        "zip_entry": "detect.tflite",
    },
]

# ─────────────────────────────────────────────────────────────
# 다운로드 유틸리티
# ─────────────────────────────────────────────────────────────

def _download_file(url: str, dest: Path, timeout: int = 30) -> bool:
    """단일 URL을 다운로드. 성공 여부 반환."""
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "TFLite-SAST-Downloader/1.0"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            chunk = 65536  # 64 KB
            with open(dest, "wb") as f:
                while True:
                    block = resp.read(chunk)
                    if not block:
                        break
                    f.write(block)
                    downloaded += len(block)
                    if total > 0:
                        pct = downloaded / total * 100
                        print(f"\r    {downloaded//1024:,} KB / {total//1024:,} KB "
                              f"({pct:.0f}%)", end="", flush=True)
        print()
        return True
    except urllib.error.URLError as e:
        print(f"\n    [오류] URL 접근 실패: {e}")
        return False
    except Exception as e:
        print(f"\n    [오류] 다운로드 실패: {e}")
        return False


def _extract_tflite_from_zip(zip_path: Path, entry: str, dest: Path) -> bool:
    """ZIP 파일에서 .tflite 파일 추출 (zipfile 표준 라이브러리)."""
    import zipfile
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            names = z.namelist()
            # entry 파일명이 경로 끝에 매치되는 항목 찾기
            matched = [n for n in names if n.endswith(entry) or n == entry]
            if not matched:
                print(f"    [경고] ZIP 내 {entry} 없음. 포함 파일: {names[:5]}")
                return False
            target = matched[0]
            data = z.read(target)
            dest.write_bytes(data)
            return True
    except Exception as e:
        print(f"    [오류] ZIP 추출 실패: {e}")
        return False


# ─────────────────────────────────────────────────────────────
# 메인 다운로드 루프
# ─────────────────────────────────────────────────────────────

def download_all(output_dir: str = "real_models", timeout: int = 30) -> Path:
    out = Path(output_dir)
    out.mkdir(exist_ok=True)

    print(f"\n[*] 실제 TFLite 모델 다운로드 시작 → {out.resolve()}")
    print(f"    (네트워크 오류 시 해당 모델은 건너뜁니다)\n")

    results = []
    success = 0

    for spec in REAL_MODELS:
        mid  = spec["id"]
        name = spec["name"]
        url  = spec["url"]
        is_zip = spec.get("is_zip", False)
        zip_entry = spec.get("zip_entry", "")

        ext = ".zip" if is_zip else ".tflite"
        dest_tmp  = out / f"{mid}{ext}"
        dest_final = out / f"{mid}.tflite"

        if dest_final.exists():
            print(f"  [SKIP] {mid}.tflite  -- 이미 존재")
            success += 1
            results.append({"id": mid, "status": "exists",
                            "name": name, "note": spec.get("note", "")})
            continue

        print(f"  [{mid}] {name}")
        print(f"    URL: {url[:72]}...")
        t0 = time.perf_counter()
        ok = _download_file(url, dest_tmp, timeout)
        elapsed = time.perf_counter() - t0

        if not ok:
            results.append({"id": mid, "status": "failed", "name": name})
            continue

        if is_zip:
            ok = _extract_tflite_from_zip(dest_tmp, zip_entry, dest_final)
            dest_tmp.unlink(missing_ok=True)
            if not ok:
                results.append({"id": mid, "status": "extract_failed", "name": name})
                continue
        else:
            dest_tmp.rename(dest_final)

        size_kb = dest_final.stat().st_size / 1024
        print(f"    [OK] {size_kb:.1f} KB  ({elapsed:.1f}s)")
        success += 1
        results.append({
            "id": mid, "status": "ok", "name": name,
            "size_kb": round(size_kb, 1),
            "note": spec.get("note", ""),
        })

    print(f"\n[*] 완료: {success}/{len(REAL_MODELS)}개 다운로드 성공")
    return out


# ─────────────────────────────────────────────────────────────
# 다운로드 후 즉시 분석 (--analyze 옵션)
# ─────────────────────────────────────────────────────────────

def analyze_real_models(model_dir: str):
    """다운로드된 실제 모델을 tflite_sast.py로 분석하여 FPR 측정."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    import tflite_sast as sast

    model_path = Path(model_dir)
    files = sorted(model_path.glob("R_*.tflite"))

    if not files:
        print("[경고] 분석할 실제 모델이 없습니다.")
        return

    print(f"\n{'='*65}")
    print(f"  실제 모델 분석 결과 (FPR 측정용)")
    print(f"{'='*65}")
    print(f"  {'Model':<12} {'Size(KB)':>8} {'Tensors':>7} "
          f"{'Ops':>5} {'Findings':>8} {'R(M)':>6}")
    print(f"  {'-'*55}")

    fp_count = 0
    total    = 0

    for fpath in files:
        try:
            data   = fpath.read_bytes()
            result = sast.analyze(data, fpath.name)
            total += 1

            findings = len(result["findings"])
            if findings > 0:
                fp_count += 1
                flag = "  ← FP 후보"
            else:
                flag = ""

            print(f"  {fpath.name:<12} "
                  f"{result['file_size']/1024:>8.1f} "
                  f"{result['tensors']:>7} "
                  f"{result['operators']:>5} "
                  f"{findings:>8} "
                  f"{result['R']:>6.4f}{flag}")

            # 발견 사항 상세 출력
            for f in result["findings"]:
                print(f"    [{f['severity']}] {f['rule']} {f['cve']}: "
                      f"{f['message'][:60]}...")
        except Exception as e:
            print(f"  {fpath.name:<12} [파싱 오류] {e}")

    print(f"  {'-'*55}")
    fpr = fp_count / total if total > 0 else 0.0
    print(f"\n  실제 모델 FPR = {fp_count}/{total} = {fpr*100:.1f}%")
    print()
    if fp_count > 0:
        print("  [!] FP 후보 발견: 해당 규칙의 조건을 검토하고")
        print("      임계값 또는 컨텍스트 조건을 강화할 것을 권장합니다.")
    else:
        print("  [OK] 실제 모델에서 오탐 없음 → 논문 FPR 수치 신뢰도 확보")

    return {"fpr": fpr, "fp": fp_count, "total": total}


# ─────────────────────────────────────────────────────────────
# 수동 다운로드 안내 (네트워크 없는 환경용)
# ─────────────────────────────────────────────────────────────

MANUAL_DOWNLOAD_GUIDE = """
네트워크가 없거나 다운로드에 실패한 경우, 아래 방법으로 수동 확보하세요:

1. PC에서 다음 URL로 .tflite 파일을 직접 다운로드:
   - https://tfhub.dev/tensorflow/lite-model/imagenet/mobilenet_v2_1.0_224/classification/5/default/1
   - https://www.kaggle.com/models/ (TFLite 모델 검색)

2. 라즈베리파이로 복사:
   scp mobilenet_v2.tflite pi@raspberrypi:/path/to/real_models/R_B02.tflite

3. tflite_sast.py로 직접 분석:
   python tflite_sast.py real_models/R_B02.tflite

참고: TFLite 모델 zoo GitHub:
   https://github.com/tensorflow/tflite-support/tree/master/tensorflow_lite_support/metadata/python/tests/testdata
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="실제 TFLite 모델 다운로더 (표준 라이브러리 전용)"
    )
    parser.add_argument("--dir", "-d", default="real_models",
                        help="저장 디렉터리 (기본: real_models)")
    parser.add_argument("--analyze", "-a", action="store_true",
                        help="다운로드 후 즉시 분석 실행")
    parser.add_argument("--timeout", type=int, default=30,
                        help="다운로드 타임아웃(초) (기본: 30)")
    parser.add_argument("--guide", action="store_true",
                        help="수동 다운로드 안내 출력")
    args = parser.parse_args()

    if args.guide:
        print(MANUAL_DOWNLOAD_GUIDE)
    else:
        out = download_all(args.dir, args.timeout)
        if args.analyze:
            analyze_real_models(args.dir)
