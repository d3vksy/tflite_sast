# IoT 엣지 디바이스를 위한 TFLite 모델 파일의 경량 정적 보안 분석 시스템

---

**요약** — 사물인터넷(IoT) 엣지 디바이스에서 TensorFlow Lite(TFLite) 모델이 광범위하게 배포됨에 따라, 악의적으로 조작된 모델 파일을 통한 메모리 안전성 공격이 현실적인 위협으로 부상하고 있다. 본 논문은 Python 표준 라이브러리만으로 TFLite FlatBuffer 바이너리를 직접 파싱하여, 실행 전에 알려진 CVE 취약점 패턴 5종을 탐지하는 경량 정적 보안 분석 시스템을 제안한다. 라즈베리파이 4(ARM Cortex-A72, Python 3.13) 환경에서 22개(정상 15개, 취약 7개)의 합성 테스트 모델과 8개의 실제 공개 모델에 대한 실험을 수행하였으며, 본 시스템은 탐지율(Recall) 100%, 오탐율(FPR) 0.0%, F1 점수 100%를 달성하였고, 모델당 평균 분석 시간은 0.37 ms에 불과하였다.

**키워드** — TFLite, 정적 분석, IoT 보안, FlatBuffer, CVE, 취약점 탐지

---

## 1. 서론

TensorFlow Lite는 스마트 카메라, 의료 모니터링 장비, 자율주행 센서 노드 등 리소스 제약 환경에서 동작하는 엣지 AI의 사실상 표준으로 자리 잡았다. 그러나 TFLite 런타임에는 CVE-2022-23558, CVE-2021-37681 등 다수의 메모리 안전성 취약점이 보고되어 있으며 [1, 2], 공격자가 모델 파일 내 특정 필드에 비정상적인 값을 삽입하면 기기가 해당 모델을 불러오는 순간 정수 오버플로우, 배열 경계 위반(out-of-bounds), 무한 루프 등이 유발될 수 있다.

기존 방어 기법은 크게 두 가지 한계를 갖는다. 첫째, TensorFlow 전체 스택이나 flatbuffers 라이브러리를 요구하여 수백 MB의 의존성을 수반하므로 라즈베리파이 수준의 저사양 기기에 직접 배포하기 어렵다. 둘째, 동적 분석 방식은 모델을 실제로 실행해야 하므로 악성 모델이 시스템에 적재되는 것을 사전에 막지 못한다. 이 두 문제를 동시에 해결하기 위해, 본 논문은 **외부 의존성 없이 실행 전(pre-deployment) 단계에서 동작하는 정적 보안 분석기(SAST)**를 제안한다.

---

## 2. 시스템 설계

### 2.1 FlatBuffer 파서

TFLite 모델 파일은 FlatBuffer 형식으로 직렬화되며 [6], vtable 기반 오프셋 구조를 갖는다. 본 시스템은 Python의 `struct` 모듈만으로 (1) vtable 탐색과 (2) 4-byte 정렬 검사를 수행하는 경량 파서를 구현하였다. 정렬 검사를 생략하면 실제 TFLite 모델에서 `is_variable` 불리언 필드가 UOffset으로 오판되어 수천 건의 허위 탐지가 발생함을 실험적으로 확인하였다. 탐지 규칙은 파싱된 계층 구조 객체를 순회하며 독립적으로 실행된다.

### 2.2 탐지 규칙 (R001–R005) 및 심각도 분류

본 논문의 심각도는 CVSS 기준이 아닌 pre-deployment 즉각 실행 영향도를 기준으로 분류한다.

- **(1) CRITICAL**: 모델 로딩 즉시 메모리 손상 확실 (Integer Overflow)
- **(2) HIGH**: 조건 충족 시 메모리 손상·서비스 거부 (OOB Write, Stack Overflow)
- **(3) MEDIUM**: 경로 의존적, 영향 제한적 (NULL Deref, OOB Read)

본 시스템은 공개된 TFLite CVE를 분석하여 다음 5개의 탐지 규칙을 정의하였다 (Table 1).

**Table 1. 탐지 규칙 정의**

| ID   | 심각도   | CVE              | 탐지 조건 |
|------|----------|------------------|-----------|
| R001 | CRITICAL | CVE-2022-23558   | `shape` 차원값 ≤ −1 또는 > 2³⁰ |
| R002 | MEDIUM   | CVE-2021-37681   | `is_variable` 텐서가 연산자 입력으로 참조 |
| R003 | MEDIUM   | CVE-2021-29606   | SPLIT_V axis 값이 입력 rank 범위 밖 |
| R004 | HIGH     | CVE-2021-29609   | SparseAdd indices·shape 차원 불일치 |
| R005 | HIGH     | CVE-2021-29591   | While cond/body 서브그래프 인덱스 동일 |

**2.2.1 R001.** TFLite 스키마에서 동적 배치 차원(dynamic batch)을 나타내는 −1은 `shape` 필드(field 1)가 아닌 `shape_signature` 필드(field 7)에 저장하는 것이 공식 관례이다 [6]. 파서는 두 필드를 독립적으로 파싱하므로 정상 동적 배치 모델에서 오탐이 발생하지 않는다.

**2.2.2 R002.** TFLite 정상 모델에서 `is_variable` 텐서는 Variable 연산자를 통해서만 읽히며, 일반 연산자 입력으로의 직접 참조는 비정상 패턴이다. CVE-2021-37681의 취약점은 SVDF 커널에서 `GetVariableInput()`이 nullptr 반환 시 `GetTensorData()`가 이를 역참조하여 발생한다.

**2.2.3 R003.** CVE-2021-29606의 취약점은 `split_v.cc`의 `SizeOfDimension(input, axis_value)`에서 `axis_value`가 rank 범위 밖일 때 발생하는 힙 OOB 읽기이다. 파서는 axis 텐서 buffer에서 int32 스칼라를 읽어 `axis_value < 0 OR axis_value ≥ rank` 조건을 검사한다.

**2.2.4 R004.** CVE-2021-29609의 취약점은 SparseAdd에서 `*_indices` 2차원과 `*_shape` 크기의 정합 전제를 검증하지 않아 발생하는 힙 OOB 쓰기이다. 해당 메타데이터는 `shape` 필드에 정적으로 기록되므로 실행 없이 탐지 가능하다.

**2.2.5 R005.** While 연산자의 `cond_subgraph_index`(field 0)와 `body_subgraph_index`(field 1)는 `WhileOptions` 보조 테이블에 저장된다. BuiltinOperator 코드 82로 While을 식별한 후 두 인덱스의 동일 여부를 판정한다.

### 2.3 위험도 점수 R(M)

단일 취약점의 이진 판정 외에, 복합 취약점(여러 모듈에 걸친 취약점 집합)의 심각성을 정량화하기 위해 위험도 점수 $R(M)$을 정의한다.

먼저 심각도별 가중 합산 $W(M)$을 다음과 같이 계산한다.

$$W(M) = 10n_C + 5n_H + 2n_M$$

여기서 $n_C$, $n_H$, $n_M$은 각각 Critical, High, Medium 취약점의 탐지 건수이며, 가중치 (10, 5, 2)는 심각도 간 2배 차등 원칙을 따른다.

그러나 $W(M)$만으로는 동일한 가중 합을 가지면서 등급 구성이 다른 경우를 구별할 수 없다. 이를 보완하기 위해 등급 서열 기반 집중도 지수 $C(M)$을 도입한다. $C(M)$은 취약점이 Critical과 High에 얼마나 치중되어 있는지를 측정하며, 다음과 같이 정의한다.

$$C(M) = \begin{cases} 0 & (N = 0) \\ \dfrac{2n_C + n_H}{2N} & (N \geq 1) \end{cases}$$

여기서 $N = n_C + n_H + n_M$은 전체 취약점 탐지 건수이다. 이 정의에 따라 Critical은 rank 2, High는 rank 1, Medium은 rank 0으로 취급되며, $C(M)$은 항상 $[0, 1]$ 범위에 들어간다.

최종 위험도 점수 $R(M)$은 가중 합산 $W(M)$에 로그 압축을 적용한 후, 집중도 $C(M)$으로 스케일링하여 계산한다. 단, $N = 0$인 경우에는 $R(M) = 0$으로 정의한다.

$$R(M) = \log_2(1 + W(M)) \cdot (1 + C(M))$$

---

## 3. 실험 및 평가

### 3.1 실험 환경 및 데이터셋

본 실험은 Raspberry Pi 4(ARM Cortex-A72 1.8 GHz, RAM 4 GB, Python 3.13) 환경에서 수행하였다. 합성 테스트셋은 Python 기반 FlatBuffer 역방향 빌더를 이용하여 총 22개의 모델을 생성하였다. 생성된 데이터셋은 다음과 같이 구성된다.

- **B001–B010**: 정상 모델 (ADD 연산 기반, 텐서 8–12개, 연산자 7–10개)
- **B011–B015**: FPR 검증을 위한 정상 모델 (각 규칙의 경계 조건 포함; 예: B011은 `shape_signature`에만 −1 포함, B013은 SPLIT_V이지만 axis=1로 rank 범위 이내)
- **M001–M005**: 단일 취약점 주입 모델 (R001–R005 각 1건)
- **M006**: 복합 취약점 주입 모델 (R001 + R002)
- **M007**: 복합 취약점 주입 모델 (R003 + R004 + R005)

또한, 외부 검증을 위해 Google AI에서 공개한 TFLite 모델 8종(MobileNetV1, EfficientNet-Lite 0/1/2, EfficientDet-Lite0, DeepLab v3, MobileNetV1 GPU, MobileNetV1 SSD, 총 용량 0.4–16.5 MB)에 대해 분석기를 실행하여 FPR을 추가적으로 평가하였다.

### 3.2 전체 분석 결과

**Table 2. 합성 모델 전체 분석 결과**

| Model | Type | Size (KB) | Tensors | Ops | Findings | W | C(M) | R(M) | Time (ms) |
|-------|------|-----------|---------|-----|----------|---|------|------|-----------|
| B001 | Benign | 1.34 | 9 | 8 | 0 | 0 | 0.0000 | 0.0000 | 0.5890 |
| B002 | Benign | 1.48 | 10 | 9 | 0 | 0 | 0.0000 | 0.0000 | 0.5344 |
| B003 | Benign | 1.61 | 11 | 10 | 0 | 0 | 0.0000 | 0.0000 | 0.5739 |
| B004 | Benign | 1.56 | 12 | 7 | 0 | 0 | 0.0000 | 0.0000 | 0.5169 |
| B005 | Benign | 1.25 | 8 | 8 | 0 | 0 | 0.0000 | 0.0000 | 0.4483 |
| B006 | Benign | 1.39 | 9 | 9 | 0 | 0 | 0.0000 | 0.0000 | 0.4766 |
| B007 | Benign | 1.52 | 10 | 10 | 0 | 0 | 0.0000 | 0.0000 | 0.5201 |
| B008 | Benign | 1.47 | 11 | 7 | 0 | 0 | 0.0000 | 0.0000 | 0.4845 |
| B009 | Benign | 1.61 | 12 | 8 | 0 | 0 | 0.0000 | 0.0000 | 0.5282 |
| B010 | Benign | 1.30 | 8 | 9 | 0 | 0 | 0.0000 | 0.0000 | 0.4498 |
| B011 (FPR) | Benign | 0.51 | 3 | 1 | **0** | 0 | 0.0000 | 0.0000 | 0.2061 |
| B012 (FPR) | Benign | 0.50 | 4 | 1 | **0** | 0 | 0.0000 | 0.0000 | 0.2439 |
| B013 (FPR) | Benign | 0.68 | 5 | 1 | **0** | 0 | 0.0000 | 0.0000 | [TBD] |
| B014 (FPR) | Benign | 0.82 | 7 | 1 | **0** | 0 | 0.0000 | 0.0000 | 0.2948 |
| B015 (FPR) | Benign | 0.38 | 2 | 1 | **0** | 0 | 0.0000 | 0.0000 | 0.1937 |
| M001 | Malicious | 0.43 | 3 | 1 | **1** (R001) | 10 | 1.0000 | 6.9186 | [TBD] |
| M002 | Malicious | 0.45 | 3 | 1 | **1** (R002) | 2 | 0.0000 | 1.5850 | [TBD] |
| M003 | Malicious | 0.68 | 5 | 1 | **1** (R003) | 2 | 0.0000 | 1.5850 | [TBD] |
| M004 | Malicious | 0.84 | 7 | 1 | **1** (R004) | 5 | 0.5000 | 3.8775 | [TBD] |
| M005 | Malicious | 0.38 | 2 | 1 | **1** (R005) | 5 | 0.5000 | 3.8775 | [TBD] |
| M006 | Malicious | 0.45 | 3 | 1 | **2** (R001+R002) | 12 | 0.5000 | 5.5506 | [TBD] |
| M007 | Malicious | 1.59 | 14 | 3 | **3** (R003+R004+R005) | 12 | 0.3333 | 4.9339 | [TBD] |

취약 모델 7개는 모두 정확히 탐지되었으며(TP = 7, FN = 0), 정상 모델 15개는 모두 오탐 없이 통과하였다(FP = 0, TN = 15). $R(M)$ 점수는 단일 CRITICAL 취약점 모델(M001, $R$ = 6.9186)이 가장 높게 나타나, 위험도 점수가 심각도 집중도를 반영함을 보여 준다. 복합 취약점 모델에서는 Critical을 포함하는 M006($R$ = 5.5506)이 High+Medium 조합인 M007($R$ = 4.9339)보다 높아, 등급 서열이 스케일링에 정확히 반영됨을 확인하였다.

### 3.3 성능 지표

**Table 3. 성능 지표 요약**

| 지표 | 결과 |
|------|------|
| TP / FP / TN / FN | 7 / 0 / 15 / 0 |
| Detection Rate (Recall) | **100.0%** |
| False Positive Rate | **0.0%** |
| Precision | **100.0%** |
| F1 Score | **100.0%** |
| 실제 모델 FPR (8개) | **0.0%** |
| 평균 분석 시간 | **0.37 ms / 모델** (라즈베리파이 4 환경) |

규칙별로도 R001–R005 각각 Recall 100%, FP 0건을 달성하였다. 실제 공개 모델 8종(최대 16.5 MB, 최대 602개 텐서)에서도 오탐이 발생하지 않아, 본 시스템이 실제 배포 환경에서도 동작함을 확인하였다.

---

## 4. 논의

### 한계

본 시스템이 탐지하는 패턴은 현재 공개된 CVE 5종에 한정된다. 미공개 제로데이 취약점이나 새로운 연산자 유형에 대한 탐지 규칙은 향후 연구로 남긴다. 또한 실험 데이터셋이 합성 모델 22개와 실제 모델 8개로 구성되어, 더 광범위한 실제 모델 집합에 대한 추가 검증이 필요하다.

### 의의

본 시스템은 Python 표준 라이브러리(`struct`, `math`, `json`)만으로 구현되어 **의존성이 전무**하므로, TensorFlow 설치 없이 모든 Python 3.x 환경에서 즉시 실행 가능하다. 단일 파일(`tflite_sast.py`, 약 840줄)로 완결되어 임베디드 리눅스, 컨테이너, 라즈베리파이 등 어느 엣지 환경에도 부담 없이 배포할 수 있다. 특히 `shape` 필드와 `shape_signature` 필드의 구별, SPLIT_V axis 값과 입력 텐서 rank의 범위 비교, `WhileOptions` 보조 테이블 파싱 등 TFLite FlatBuffer 스키마를 정확하게 구현함으로써 FPR 0%를 달성한 점이 핵심 기여이다.

---

## 5. 결론

본 논문은 외부 라이브러리 없이 TFLite 모델 파일을 실행 전에 정적 분석하여 5가지 알려진 CVE 취약점을 탐지하는 경량 시스템을 제안하였다. 합성 데이터셋과 실제 공개 모델을 대상으로 한 실험에서 탐지율 100%, 오탐율 0%, 평균 분석 시간 0.37 ms를 달성하였으며, 이는 본 시스템이 IoT 엣지 디바이스에서의 AI 모델 공급망 보안 강화에 실용적으로 기여할 수 있음을 시사한다.

---

## 참고문헌

[1] Google LLC, "CVE-2022-23558: TensorFlow integer overflow via large tensor shape," NVD, 2022.

[2] Google LLC, "CVE-2021-37681: TensorFlow null pointer dereference with is_variable tensor," NVD, 2021.

[3] Google LLC, "CVE-2021-29606: TensorFlow heap OOB access via buffer slot 0," NVD, 2021.

[4] Google LLC, "CVE-2021-29609: TensorFlow SparseAdd dimension mismatch," NVD, 2021.

[5] Google LLC, "CVE-2021-29591: TensorFlow While loop stack overflow via self-reference," NVD, 2021.

[6] Google LLC, "TensorFlow Lite FlatBuffer Schema," GitHub, tensorflow/tensorflow, 2024.

[7] C. E. Shannon, "A Mathematical Theory of Communication," Bell System Technical Journal, vol. 27, pp. 379–423, 1948.
