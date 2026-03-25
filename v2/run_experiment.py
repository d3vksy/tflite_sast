#!/usr/bin/env python3
"""
run_experiment.py — TFLite 정적 보안 분석 실험 자동화 스크립트
테스트 모델 생성 → 분석 → Table 3 형식 출력 → Detection Rate/FPR 계산 → 결과 저장
"""

import time
import json
import os
import sys
from pathlib import Path

# 같은 디렉터리의 모듈을 import
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import generate_test_models as gen
import tflite_sast as sast

# ──────────────────────────────────────────────
# 상수 및 설정
# ──────────────────────────────────────────────

MODEL_DIR    = "test_models"
RESULTS_FILE = "results.json"

# 각 모델의 실제 취약점 유무 (Ground Truth)
# B001-B010: 취약점 없음(0), M001-M007: 취약점 있음(1)
GROUND_TRUTH = {
    "B001.tflite": 0, "B002.tflite": 0, "B003.tflite": 0,
    "B004.tflite": 0, "B005.tflite": 0, "B006.tflite": 0,
    "B007.tflite": 0, "B008.tflite": 0, "B009.tflite": 0,
    "B010.tflite": 0,
    # FPR 측정용 유사 패턴 정상 모델 (취약점 없음 = 0)
    "B011.tflite": 0,   # 동적 shape -1 (정상)
    "B012.tflite": 0,   # is_variable 텐서 미참조 (정상)
    "B013.tflite": 0,   # SPLIT_V axis 정상 범위 (정상)
    "B014.tflite": 0,   # SparseAdd 차원 일치 (정상)
    "B015.tflite": 0,   # While 다른 서브그래프 (정상)
    "M001.tflite": 1, "M002.tflite": 1, "M003.tflite": 1,
    "M004.tflite": 1, "M005.tflite": 1, "M006.tflite": 1,
    "M007.tflite": 1,
}

# B011-B015의 각 규칙별 "예상 오탐 여부" 메모
# (True = 현재 규칙으로 FP 발생이 예상됨 → 규칙 개선 필요)
FP_RISK = {
    "B011.tflite": {"R001": True,  "note": "동적 shape -1을 취약점으로 오인"},
    "B012.tflite": {"R002": False, "note": "미참조 is_variable → 정상 탐지"},
    # [수정] R003 탐지 조건이 'SPLIT_V axis 범위 위반'으로 변경됨에 따라 주석 갱신
    "B013.tflite": {"R003": False, "note": "SPLIT_V axis 정상 범위(0 ≤ axis < rank) → 정상 탐지"},
    "B014.tflite": {"R004": False, "note": "차원 일치 → 정상 탐지"},
    "B015.tflite": {"R005": False, "note": "다른 인덱스 → 정상 탐지"},
}

# 모델 타입 라벨
MODEL_TYPES = {
    **{f"B{i:03d}.tflite": "Benign"     for i in range(1, 16)},
    **{f"M{i:03d}.tflite": "Malicious"  for i in range(1, 8)},
}

# 예상 취약점 규칙 (모델별 기대 탐지 규칙)
EXPECTED_RULES = {
    "M001.tflite": {"R001"},
    "M002.tflite": {"R002"},
    "M003.tflite": {"R003"},
    "M004.tflite": {"R004"},
    "M005.tflite": {"R005"},
    "M006.tflite": {"R001", "R002"},
    "M007.tflite": {"R003", "R004", "R005"},
}

# ANSI 컬러
CYAN   = "\033[96m"
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

# ──────────────────────────────────────────────
# 단계 1: 테스트 모델 생성
# ──────────────────────────────────────────────

def step1_generate_models(model_dir: str):
    """generate_test_models.py를 호출하여 22개 모델 생성"""
    print(f"\n{BOLD}{'='*65}{RESET}")
    print(f"{BOLD}  단계 1: 테스트 모델 생성{RESET}")
    print(f"{'='*65}")
    gen.generate_all(model_dir)


# ──────────────────────────────────────────────
# 단계 2: 각 모델 분석 (시간 측정)
# ──────────────────────────────────────────────

def step2_analyze_models(model_dir: str) -> list:
    """모든 .tflite 파일을 분석하고 결과 목록 반환"""
    print(f"\n{BOLD}{'='*65}{RESET}")
    print(f"{BOLD}  단계 2: 모델 분석 실행{RESET}")
    print(f"{'='*65}")

    model_path = Path(model_dir)
    # 파일명 정렬: B001-B015, M001-M007 순서 보장
    all_files = sorted(model_path.glob("*.tflite"),
                       key=lambda p: p.name)

    results = []
    for fpath in all_files:
        data = fpath.read_bytes()

        # 분석 시간 측정
        t_start = time.perf_counter()
        result  = sast.analyze(data, fpath.name)
        t_end   = time.perf_counter()

        elapsed = t_end - t_start
        result["elapsed"]    = elapsed
        result["model_type"] = MODEL_TYPES.get(fpath.name, "Unknown")

        # [수정] set → sorted list 변환: 재현성 보장 및 results.json diff 노이즈 제거
        detected_rules = {f["rule"] for f in result["findings"]}
        result["detected_rules"] = list(sorted(detected_rules))

        # 탐지 여부 (findings가 1개 이상이면 탐지)
        result["detected"] = len(result["findings"]) > 0

        print(f"  {fpath.name:16s}  findings={len(result['findings'])}  "
              f"R(M)={result['R']:.4f}  time={elapsed*1000:.3f}ms")

        results.append(result)

    return results


# ──────────────────────────────────────────────
# 단계 3: Table 3 형식 출력
# ──────────────────────────────────────────────

def step3_print_table3(results: list):
    """
    Table 3 형식으로 결과 출력
    Model | Type | Size(KB) | Tensors | Ops | Findings | W | C(M) | R(M) | Time(ms)
    """
    print(f"\n{BOLD}{'='*65}{RESET}")
    print(f"{BOLD}  Table 3: 분석 결과 요약{RESET}")
    print(f"{'='*65}")

    header = (
        f"{'Model':<10} {'Type':<10} {'Size(KB)':>8} "
        f"{'Tensors':>7} {'Ops':>5} {'Findings':>8} "
        f"{'W':>5} {'C(M)':>6} {'R(M)':>6} {'Time(ms)':>9}"
    )
    sep = "-" * len(header)
    print(f"{BOLD}{header}{RESET}")
    print(sep)

    for r in results:
        fname    = r["filename"]
        mtype    = r["model_type"]
        size_kb  = r["file_size"] / 1024.0
        tensors  = r["tensors"]
        ops      = r["operators"]
        findings = len(r["findings"])
        W        = r["W"]
        C        = r["C"]
        R        = r["R"]
        elapsed  = r["elapsed"] * 1000  # ms

        line_color = YELLOW if mtype == "Malicious" else ""
        line = (
            f"{fname:<10} {mtype:<10} {size_kb:>8.2f} "
            f"{tensors:>7} {ops:>5} {findings:>8} "
            f"{W:>5.0f} {C:>6.4f} {R:>6.4f} {elapsed:>9.3f}"
        )
        print(f"{line_color}{line}{RESET}")

    print(sep)


# ──────────────────────────────────────────────
# 단계 4: Detection Rate 및 FPR 계산
# ──────────────────────────────────────────────

def step4_compute_metrics(results: list):
    """
    이진 분류 성능 지표 계산
    - Detection Rate (Recall) = TP / (TP + FN)
    - False Positive Rate     = FP / (FP + TN)
    - Precision               = TP / (TP + FP)
    - F1 Score                = 2 * P * R / (P + R)
    - 규칙별 탐지 정확도
    """
    print(f"\n{BOLD}{'='*65}{RESET}")
    print(f"{BOLD}  단계 4: 탐지 성능 평가{RESET}")
    print(f"{'='*65}")

    TP = FP = TN = FN = 0
    rule_stats = {}  # rule_id → {TP, FN}

    for r in results:
        fname    = r["filename"]
        gt       = GROUND_TRUTH.get(fname, -1)
        detected = r["detected"]

        if gt == 1 and detected:
            TP += 1
        elif gt == 0 and detected:
            FP += 1
        elif gt == 0 and not detected:
            TN += 1
        elif gt == 1 and not detected:
            FN += 1

        # 규칙별 탐지 정확도
        if fname in EXPECTED_RULES:
            exp = EXPECTED_RULES[fname]
            det = set(r["detected_rules"])
            for rule in exp:
                if rule not in rule_stats:
                    rule_stats[rule] = {"TP": 0, "FN": 0}
                if rule in det:
                    rule_stats[rule]["TP"] += 1
                else:
                    rule_stats[rule]["FN"] += 1

    # 지표 계산
    total_pos = TP + FN
    total_neg = TN + FP

    recall    = TP / total_pos if total_pos > 0 else 0.0
    fpr       = FP / total_neg if total_neg > 0 else 0.0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    print(f"  혼동 행렬")
    print(f"    TP={TP}  FP={FP}  TN={TN}  FN={FN}")
    print()
    print(f"  {'Detection Rate (Recall)':30s}: {GREEN}{recall*100:.1f}%{RESET}")
    print(f"  {'False Positive Rate (FPR)':30s}: {RED}{fpr*100:.1f}%{RESET}")
    print(f"  {'Precision':30s}: {precision*100:.1f}%")
    print(f"  {'F1 Score':30s}: {f1*100:.1f}%")
    print()

    print(f"  규칙별 탐지 현황")
    print(f"  {'Rule':<8} {'TP':>4} {'FN':>4} {'Recall':>8}")
    print(f"  {'-'*28}")
    for rule in sorted(rule_stats.keys()):
        s = rule_stats[rule]
        tp = s["TP"]
        fn = s["FN"]
        r_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        status = f"{GREEN}OK{RESET}" if r_recall == 1.0 else f"{RED}NG{RESET}"
        print(f"  {rule:<8} {tp:>4} {fn:>4} {r_recall*100:>7.1f}% {status}")

    return {
        "TP": TP, "FP": FP, "TN": TN, "FN": FN,
        "recall":    recall,
        "fpr":       fpr,
        "precision": precision,
        "f1":        f1,
        "rule_stats": rule_stats,
    }


# ──────────────────────────────────────────────
# 단계 5: 결과 저장
# ──────────────────────────────────────────────

def step5_save_results(results: list, metrics: dict, output_file: str):
    """분석 결과를 results.json으로 저장"""
    print(f"\n{BOLD}{'='*65}{RESET}")
    print(f"{BOLD}  단계 5: 결과 저장{RESET}")
    print(f"{'='*65}")

    def _clean(obj):
        if isinstance(obj, bytes):
            return obj.hex()
        if isinstance(obj, set):
            return sorted(obj)   # set도 정렬하여 저장
        return obj

    output = {
        "experiment": "TFLite 정적 보안 분석 시스템 — 논문 실험 재현",
        "total_models": len(results),
        "metrics": metrics,
        "models": [],
    }

    for r in results:
        entry = {
            "filename":       r["filename"],
            "model_type":     r["model_type"],
            "file_size_kb":   round(r["file_size"] / 1024.0, 2),
            "version":        r["version"],
            "subgraphs":      r["subgraphs"],
            "tensors":        r["tensors"],
            "operators":      r["operators"],
            "buffers":        r["buffers"],
            "findings_count": len(r["findings"]),
            "detected_rules": r["detected_rules"],  # 이미 sorted list
            "W":              round(r["W"], 4),
            "C":              round(r["C"], 4),
            "R":              round(r["R"], 4),
            "elapsed_ms":     round(r["elapsed"] * 1000, 4),
            "ground_truth":   GROUND_TRUTH.get(r["filename"], -1),
            "detected":       r["detected"],
            "findings": [
                {
                    "rule":     f["rule"],
                    "severity": f["severity"],
                    "cve":      f["cve"],
                    "message":  f["message"],
                }
                for f in r["findings"]
            ],
        }
        output["models"].append(entry)

    out_path = Path(output_file)
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(output, fp, ensure_ascii=False, indent=2, default=_clean)

    print(f"  결과 저장 완료: {out_path.resolve()}")
    print(f"  총 {len(results)}개 모델, "
          f"{sum(len(r['findings']) for r in results)}건 발견 사항")


# ──────────────────────────────────────────────
# 실험 요약 출력
# ──────────────────────────────────────────────

def print_summary(results: list, metrics: dict):
    """실험 전체 요약 출력"""
    total_time = sum(r["elapsed"] for r in results)
    avg_time   = total_time / len(results) if results else 0

    print(f"\n{BOLD}{'='*65}{RESET}")
    print(f"{BOLD}  실험 요약{RESET}")
    print(f"{'='*65}")
    print(f"  분석 모델 수   : {len(results)}개")
    print(f"  총 소요 시간   : {total_time*1000:.2f} ms")
    print(f"  평균 분석 시간 : {avg_time*1000:.3f} ms / 모델")
    print(f"  Detection Rate : {metrics['recall']*100:.1f}%")
    print(f"  FPR            : {metrics['fpr']*100:.1f}%")
    print(f"  F1 Score       : {metrics['f1']*100:.1f}%")
    print(f"{'='*65}\n")


# ──────────────────────────────────────────────
# 메인 진입점
# ──────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="TFLite 정적 보안 분석 실험 자동화 스크립트"
    )
    parser.add_argument(
        "--model-dir", "-d",
        default=MODEL_DIR,
        help=f"테스트 모델 디렉터리 (기본: {MODEL_DIR})"
    )
    parser.add_argument(
        "--output", "-o",
        default=RESULTS_FILE,
        help=f"결과 JSON 파일 경로 (기본: {RESULTS_FILE})"
    )
    parser.add_argument(
        "--skip-generate",
        action="store_true",
        help="모델 생성 단계 건너뜀 (이미 생성된 경우)"
    )
    args = parser.parse_args()

    print(f"{BOLD}{CYAN}")
    print("  +----------------------------------------------------------+")
    print("  |   TFLite 모델 정적 보안 분석 시스템 -- 실험 재현        |")
    print("  |   IoT 엣지 디바이스 보안 분석 논문 구현                 |")
    print("  +----------------------------------------------------------+")
    print(f"{RESET}")

    # 단계 1: 모델 생성
    if not args.skip_generate:
        step1_generate_models(args.model_dir)
    else:
        print(f"[*] 모델 생성 건너뜀 (--skip-generate)")

    # 단계 2: 분석
    results = step2_analyze_models(args.model_dir)

    if not results:
        print(f"{RED}[오류] 분석할 모델이 없습니다. "
              f"{args.model_dir} 디렉터리를 확인하세요.{RESET}")
        sys.exit(1)

    # 단계 3: Table 3 출력
    step3_print_table3(results)

    # 단계 4: 지표 계산
    metrics = step4_compute_metrics(results)

    # 단계 5: 저장
    step5_save_results(results, metrics, args.output)

    # 요약
    print_summary(results, metrics)


if __name__ == "__main__":
    main()
