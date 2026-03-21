#!/usr/bin/env python3
"""
tflite_sast.py — TFLite 모델 정적 보안 분석기
FlatBuffer를 순수 Python struct 모듈로 직접 파싱하여 취약점을 탐지한다.
"""

import struct
import math
import json
import os
import argparse
from pathlib import Path

# ──────────────────────────────────────────────
# ANSI 컬러 코드
# ──────────────────────────────────────────────
ANSI_RED    = "\033[91m"
ANSI_ORANGE = "\033[38;5;208m"
ANSI_YELLOW = "\033[93m"
ANSI_WHITE  = "\033[97m"
ANSI_RESET  = "\033[0m"
ANSI_BOLD   = "\033[1m"
ANSI_CYAN   = "\033[96m"
ANSI_GREEN  = "\033[92m"

SEVERITY_COLOR = {
    "CRITICAL": ANSI_RED,
    "HIGH":     ANSI_ORANGE,
    "MEDIUM":   ANSI_YELLOW,
    "LOW":      ANSI_WHITE,
}

SEVERITY_WEIGHT = {"CRITICAL": 10, "HIGH": 5, "MEDIUM": 2, "LOW": 1}

# BuiltinOperator 상수 (schema.fbs 기준)
BUILTIN_WHILE = 82
BUILTIN_ADD   = 0
BUILTIN_SPARSE_TO_DENSE = 100

# ──────────────────────────────────────────────
# FlatBuffer 저수준 유틸리티
# ──────────────────────────────────────────────

def _read_u8(buf, offset):
    """1바이트 부호 없는 정수 읽기"""
    return struct.unpack_from("<B", buf, offset)[0]

def _read_i8(buf, offset):
    """1바이트 부호 있는 정수 읽기"""
    return struct.unpack_from("<b", buf, offset)[0]

def _read_u16(buf, offset):
    """2바이트 부호 없는 정수 읽기"""
    return struct.unpack_from("<H", buf, offset)[0]

def _read_i32(buf, offset):
    """4바이트 부호 있는 정수 읽기"""
    return struct.unpack_from("<i", buf, offset)[0]

def _read_u32(buf, offset):
    """4바이트 부호 없는 정수 읽기"""
    return struct.unpack_from("<I", buf, offset)[0]

def _read_soffset(buf, offset):
    """FlatBuffer SOffset(상대 오프셋) 읽기 — 테이블/벡터 시작 위치 계산"""
    raw = _read_i32(buf, offset)
    return offset + raw

def _read_uoffset(buf, offset):
    """FlatBuffer UOffset(절대 오프셋) 읽기 — 루트 오브젝트 위치 계산"""
    raw = _read_u32(buf, offset)
    return offset + raw

def _vtable_field(buf, table_offset, field_id):
    """
    FlatBuffer vtable에서 field_id(필드 번호)에 해당하는
    데이터 오프셋을 반환한다. 필드가 없으면 None.
    field_id: 4부터 2씩 증가 (id=0 → offset 4, id=1 → offset 6, ...)
    """
    vtable_offset = table_offset - _read_i32(buf, table_offset)
    vtable_size   = _read_u16(buf, vtable_offset)
    field_byte    = 4 + field_id * 2  # 필드 슬롯 위치 (바이트 단위)
    if field_byte >= vtable_size:
        return None
    rel = _read_u16(buf, vtable_offset + field_byte)
    if rel == 0:
        return None
    return table_offset + rel

def _read_vector(buf, table_offset, field_id):
    """
    FlatBuffer 벡터 필드를 읽어 (요소 개수, 데이터 시작 오프셋)을 반환한다.
    벡터가 없으면 (0, None) 반환.
    실제 TFLite 모델에서 비정렬 UOffset이 가중치 데이터 영역을 가리킬 수 있으므로
    bounds 검사를 포함한다.
    """
    field_off = _vtable_field(buf, table_offset, field_id)
    if field_off is None:
        return 0, None
    if field_off + 4 > len(buf):
        return 0, None
    # 4-byte UOffset은 반드시 4-byte 정렬이어야 한다.
    # 비정렬 위치는 bool 등 다른 타입의 필드이므로 absent로 처리한다.
    if field_off % 4 != 0:
        return 0, None
    raw = _read_u32(buf, field_off)
    vec_start = field_off + raw
    if vec_start + 4 > len(buf):
        return 0, None
    count = _read_u32(buf, vec_start)
    # 벡터 크기 sanity check: 실제 버퍼 크기를 초과할 수 없음
    MAX_REASONABLE = len(buf) // 4
    if count > MAX_REASONABLE:
        return 0, None
    data_end = vec_start + 4 + count * 4
    if data_end > len(buf):
        return 0, None
    return count, vec_start + 4

def _read_vector_of_tables(buf, table_offset, field_id):
    """
    벡터 내 각 요소가 테이블(오프셋 기반 참조)인 경우
    각 테이블의 절대 오프셋 리스트를 반환한다.
    """
    count, data_start = _read_vector(buf, table_offset, field_id)
    if count == 0 or data_start is None:
        return []
    result = []
    for i in range(count):
        elem_off = data_start + i * 4
        if elem_off + 4 > len(buf):
            break
        raw = _read_u32(buf, elem_off)
        tbl = elem_off + raw
        if tbl >= len(buf):
            continue
        result.append(tbl)
    return result

def _read_vector_of_i32(buf, table_offset, field_id):
    """int32 배열 벡터를 읽어 Python 리스트로 반환"""
    count, data_start = _read_vector(buf, table_offset, field_id)
    if count == 0 or data_start is None:
        return []
    return [_read_i32(buf, data_start + i * 4) for i in range(count)]

def _read_scalar_i32(buf, table_offset, field_id, default=0):
    """스칼라 int32 필드 읽기"""
    off = _vtable_field(buf, table_offset, field_id)
    if off is None:
        return default
    return _read_i32(buf, off)

def _read_scalar_u32(buf, table_offset, field_id, default=0):
    """스칼라 uint32 필드 읽기"""
    off = _vtable_field(buf, table_offset, field_id)
    if off is None:
        return default
    return _read_u32(buf, off)

def _read_scalar_u8(buf, table_offset, field_id, default=0):
    """스칼라 uint8(bool 포함) 필드 읽기"""
    off = _vtable_field(buf, table_offset, field_id)
    if off is None:
        return default
    return _read_u8(buf, off)

def _read_string(buf, table_offset, field_id):
    """FlatBuffer 문자열 필드 읽기 — UTF-8 디코딩"""
    off = _vtable_field(buf, table_offset, field_id)
    if off is None:
        return ""
    str_start = _read_uoffset(buf, off)
    length    = _read_u32(buf, str_start)
    raw_bytes = buf[str_start + 4: str_start + 4 + length]
    try:
        return raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return raw_bytes.decode("latin-1")

# ──────────────────────────────────────────────
# TFLite FlatBuffer 파서
# ──────────────────────────────────────────────

# TFLite FlatBuffer 파일 식별자 (schema.fbs file_identifier = "TFL3")
TFLITE_FILE_IDENTIFIER = b"TFL3"

class TFLiteTensor:
    """TFLite 텐서 데이터 구조"""
    __slots__ = ("name", "shape", "tensor_type", "buffer_idx", "is_variable",
                 "shape_signature")

    def __init__(self):
        self.name            = ""
        self.shape           = []   # int32 배열 — 정적 shape, 음수는 취약점
        self.tensor_type     = 0
        self.buffer_idx      = 0
        self.is_variable     = False
        self.shape_signature = []   # int32 배열 — 동적 차원은 -1로 표기 (정상)

class TFLiteOperator:
    """TFLite 연산자 데이터 구조"""
    __slots__ = ("opcode_index", "inputs", "outputs", "builtin_options_type",
                 "while_cond_subgraph", "while_body_subgraph")

    def __init__(self):
        self.opcode_index         = 0
        self.inputs               = []    # int32 배열
        self.outputs              = []    # int32 배열
        self.builtin_options_type = 0
        self.while_cond_subgraph  = None  # WhileOptions field 0 (파싱 성공 시 int)
        self.while_body_subgraph  = None  # WhileOptions field 1 (파싱 성공 시 int)

class TFLiteOperatorCode:
    """TFLite 연산자 코드 데이터 구조"""
    __slots__ = ("builtin_code", "custom_code", "deprecated_builtin_code")

    def __init__(self):
        self.builtin_code           = 0
        self.custom_code            = ""
        self.deprecated_builtin_code = 0

class TFLiteSubGraph:
    """TFLite 서브그래프 데이터 구조"""
    __slots__ = ("name", "tensors", "operators", "inputs", "outputs")

    def __init__(self):
        self.name      = ""
        self.tensors   = []
        self.operators = []
        self.inputs    = []
        self.outputs   = []

class TFLiteModel:
    """TFLite 모델 최상위 데이터 구조"""
    __slots__ = ("version", "operator_codes", "subgraphs", "buffers")

    def __init__(self):
        self.version        = 0
        self.operator_codes = []
        self.subgraphs      = []
        self.buffers        = []  # 버퍼 개수 (크기 측정용)


def _parse_tensor(buf, tbl):
    """Tensor 테이블 파싱
    실제 TFLite schema.fbs 기준 (tensorflow/lite/schema/schema.fbs):
      field 0 = name(string)
      field 1 = shape(int32[])           ← 정적 shape, 음수는 취약점
      field 2 = type(TensorType)
      field 3 = buffer(uint32)
      field 4 = quantization(QuantizationParameters)
      field 5 = is_variable(bool)        ← 실제 스키마 field 5
      field 6 = sparsity(SparsityParameters)
      field 7 = shape_signature(int32[]) ← 동적 차원은 -1, 정상 관례 (실제 스키마 field 7)
    """
    t = TFLiteTensor()
    t.name            = _read_string(buf, tbl, 0)           # field 0
    t.shape           = _read_vector_of_i32(buf, tbl, 1)    # field 1
    t.tensor_type     = _read_scalar_i32(buf, tbl, 2, 0)    # field 2
    t.buffer_idx      = _read_scalar_u32(buf, tbl, 3, 0)    # field 3
    t.is_variable     = bool(_read_scalar_u8(buf, tbl, 5, 0))  # field 5 (실제 스키마)
    t.shape_signature = _read_vector_of_i32(buf, tbl, 7)    # field 7 (실제 스키마)
    return t


def _parse_operator(buf, tbl):
    """Operator 테이블 파싱
    실제 TFLite schema.fbs 기준:
      field 0: opcode_index(uint32)
      field 1: inputs(int32[])
      field 2: outputs(int32[])
      field 3: builtin_options_type(u8 enum)
      field 4: builtin_options(union table UOffset)
      field 5: custom_options([ubyte])
    """
    op = TFLiteOperator()
    op.opcode_index = _read_scalar_u32(buf, tbl, 0, 0)   # field 0
    op.inputs       = _read_vector_of_i32(buf, tbl, 1)   # field 1
    op.outputs      = _read_vector_of_i32(buf, tbl, 2)   # field 2

    # field 3: builtin_options_type (u8 enum tag)
    op.builtin_options_type = _read_scalar_u8(buf, tbl, 3, 0)

    # field 4: builtin_options (UOffset → WhileOptions 테이블, type=119일 때)
    # WhileOptions: field 0=cond_subgraph_index(i32), field 1=body_subgraph_index(i32)
    if op.builtin_options_type == 119:  # BuiltinOptions.WhileOptions
        try:
            bo_pos = _vtable_field(buf, tbl, 4)
            if bo_pos is not None and bo_pos + 4 <= len(buf) and bo_pos % 4 == 0:
                bo_tbl = _read_uoffset(buf, bo_pos)
                if bo_tbl < len(buf):
                    op.while_cond_subgraph = _read_scalar_i32(buf, bo_tbl, 0, 0)
                    op.while_body_subgraph = _read_scalar_i32(buf, bo_tbl, 1, 0)
        except Exception:
            pass

    return op


def _parse_operator_code(buf, tbl):
    """OperatorCode 테이블 파싱
    schema 필드:
      0: deprecated_builtin_code(int8), 1: custom_code(string),
      2: version(int32), 3: builtin_code(BuiltinOperator=int32)
    """
    oc = TFLiteOperatorCode()
    oc.deprecated_builtin_code = _read_scalar_u8(buf, tbl, 0, 0)  # field 0 (int8)
    oc.custom_code              = _read_string(buf, tbl, 1)         # field 1
    oc.builtin_code             = _read_scalar_i32(buf, tbl, 3, 0) # field 3
    # deprecated_builtin_code가 127이면 builtin_code 사용, 아니면 deprecated 사용
    if oc.builtin_code == 0 and oc.deprecated_builtin_code != 127:
        oc.builtin_code = oc.deprecated_builtin_code
    return oc


def _parse_subgraph(buf, tbl):
    """SubGraph 테이블 파싱
    schema 필드:
      0: tensors(Tensor[]), 1: inputs(int32[]), 2: outputs(int32[]),
      3: operators(Operator[]), 4: name(string)
    """
    sg = TFLiteSubGraph()
    sg.name = _read_string(buf, tbl, 4)   # field 4: name

    # 텐서 벡터 파싱
    for tbl_t in _read_vector_of_tables(buf, tbl, 0):
        sg.tensors.append(_parse_tensor(buf, tbl_t))

    sg.inputs  = _read_vector_of_i32(buf, tbl, 1)   # field 1: inputs
    sg.outputs = _read_vector_of_i32(buf, tbl, 2)   # field 2: outputs

    # 연산자 벡터 파싱
    for tbl_op in _read_vector_of_tables(buf, tbl, 3):
        sg.operators.append(_parse_operator(buf, tbl_op))

    return sg


def parse_tflite(data: bytes) -> TFLiteModel:
    """
    TFLite FlatBuffer 바이너리를 파싱하여 TFLiteModel 객체를 반환한다.
    파일 식별자(TFL3)를 검증하고, Model 테이블의 모든 필드를 순회한다.
    """
    if len(data) < 8:
        raise ValueError("파일이 너무 짧습니다 (최소 8바이트 필요)")

    # 파일 식별자 검증 (오프셋 4~8)
    file_id = data[4:8]
    if file_id != TFLITE_FILE_IDENTIFIER:
        # 식별자가 없어도 파싱 시도 (일부 구버전 모델 호환)
        pass

    # 루트 오프셋 읽기
    root = _read_uoffset(data, 0)  # data[0:4] = root table offset

    model = TFLiteModel()

    # Model 필드:
    # 0: version(uint32), 1: operator_codes(OperatorCode[]),
    # 2: subgraphs(SubGraph[]), 3: description(string),
    # 4: buffers(Buffer[]), ...
    model.version = _read_scalar_u32(data, root, 0)  # field 0: version

    # operator_codes 벡터 파싱 (field 1)
    for tbl_oc in _read_vector_of_tables(data, root, 1):
        model.operator_codes.append(_parse_operator_code(data, tbl_oc))

    # subgraphs 벡터 파싱 (field 2)
    for tbl_sg in _read_vector_of_tables(data, root, 2):
        model.subgraphs.append(_parse_subgraph(data, tbl_sg))

    # buffers 개수만 파악 (field 4)
    buf_count, _ = _read_vector(data, root, 4)
    model.buffers = list(range(buf_count))  # 인덱스 목록으로 저장

    return model

# ──────────────────────────────────────────────
# 탐지 규칙 구현
# ──────────────────────────────────────────────

def _check_r001(model: TFLiteModel):
    """
    R001 (CRITICAL) CVE-2022-23558
    Tensor의 shape 차원값이 2^30 초과 또는 음수인 경우.

    주의: TFLite에서 동적 차원(-1)은 shape 필드가 아닌 shape_signature 필드에 저장된다.
    shape 필드에 음수가 나타나는 것은 정상적인 TFLite 관행이 아니며 취약점 패턴이다.
    본 파서는 shape_signature를 별도로 파싱하므로 shape_signature의 -1은 탐지하지 않는다.
    """
    findings = []
    THRESHOLD = (1 << 30)  # 2^30

    for sg_idx, sg in enumerate(model.subgraphs):
        for t_idx, tensor in enumerate(sg.tensors):
            for dim_idx, dim in enumerate(tensor.shape):
                if dim < 0 or dim > THRESHOLD:
                    findings.append({
                        "rule":     "R001",
                        "severity": "CRITICAL",
                        "cve":      "CVE-2022-23558",
                        "message":  (
                            f"서브그래프[{sg_idx}] 텐서[{t_idx}]({tensor.name!r}) "
                            f"shape[{dim_idx}]={dim} — "
                            f"비정상 차원값 (음수 또는 2^30 초과)"
                        ),
                    })
    return findings


def _check_r002(model: TFLiteModel):
    """
    R002 (HIGH) CVE-2021-37681
    is_variable=true인 텐서가 연산자 입력으로 직접 참조되는 경우
    """
    findings = []

    for sg_idx, sg in enumerate(model.subgraphs):
        # is_variable=True인 텐서 인덱스 집합
        var_indices = {
            i for i, t in enumerate(sg.tensors) if t.is_variable
        }
        if not var_indices:
            continue

        for op_idx, op in enumerate(sg.operators):
            for inp in op.inputs:
                if inp in var_indices:
                    findings.append({
                        "rule":     "R002",
                        "severity": "HIGH",
                        "cve":      "CVE-2021-37681",
                        "message":  (
                            f"서브그래프[{sg_idx}] 연산자[{op_idx}] 입력에 "
                            f"가변 텐서[{inp}]({sg.tensors[inp].name!r}) 참조"
                        ),
                    })
    return findings


def _check_r003(model: TFLiteModel):
    """
    R003 (HIGH) CVE-2021-29606
    buffer_idx=0인 상수 버퍼가 연산자 출력 텐서로 지정된 경우
    (buffer 0은 항상 빈 버퍼여야 하는 특수 슬롯)
    """
    findings = []

    for sg_idx, sg in enumerate(model.subgraphs):
        # buffer_idx=0인 텐서 인덱스
        buf0_tensors = {
            i for i, t in enumerate(sg.tensors) if t.buffer_idx == 0
        }
        if not buf0_tensors:
            continue

        for op_idx, op in enumerate(sg.operators):
            for out in op.outputs:
                if out in buf0_tensors:
                    findings.append({
                        "rule":     "R003",
                        "severity": "HIGH",
                        "cve":      "CVE-2021-29606",
                        "message":  (
                            f"서브그래프[{sg_idx}] 연산자[{op_idx}] 출력 텐서[{out}]의 "
                            f"buffer_idx=0 (예약 버퍼 슬롯 오용)"
                        ),
                    })
    return findings


def _check_r004(model: TFLiteModel):
    """
    R004 (MEDIUM) CVE-2021-29609
    SparseAdd 연산자에서 *_indices 2차원과 *_shape 크기가 불일치하는 경우
    SparseAdd는 custom_code="SparseAdd"로 식별
    입력 순서: [a_values, a_indices, a_shape, b_values, b_indices, b_shape, ...]
    a_indices: input[1], a_shape: input[2], b_indices: input[4], b_shape: input[5]
    """
    findings = []

    # SparseAdd opcode 인덱스 수집
    sparse_add_opcodes = {
        i for i, oc in enumerate(model.operator_codes)
        if oc.custom_code == "SparseAdd"
    }
    if not sparse_add_opcodes:
        return findings

    for sg_idx, sg in enumerate(model.subgraphs):
        tensors = sg.tensors
        for op_idx, op in enumerate(sg.operators):
            if op.opcode_index not in sparse_add_opcodes:
                continue
            if len(op.inputs) < 6:
                continue

            def _get_shape(tidx):
                if tidx < 0 or tidx >= len(tensors):
                    return []
                return tensors[tidx].shape

            # a_indices(rank 2 필요) vs a_shape 크기
            a_idx_shape = _get_shape(op.inputs[1])   # a_indices 텐서 shape
            a_shp_shape = _get_shape(op.inputs[2])   # a_shape 텐서 shape

            if len(a_idx_shape) >= 2 and len(a_shp_shape) >= 1:
                if a_idx_shape[1] != a_shp_shape[0]:
                    findings.append({
                        "rule":     "R004",
                        "severity": "MEDIUM",
                        "cve":      "CVE-2021-29609",
                        "message":  (
                            f"서브그래프[{sg_idx}] SparseAdd 연산자[{op_idx}] "
                            f"a_indices.shape[1]={a_idx_shape[1]} != "
                            f"a_shape.shape[0]={a_shp_shape[0]} (차원 불일치)"
                        ),
                    })

            # b_indices vs b_shape 동일 검사
            b_idx_shape = _get_shape(op.inputs[4])
            b_shp_shape = _get_shape(op.inputs[5])

            if len(b_idx_shape) >= 2 and len(b_shp_shape) >= 1:
                if b_idx_shape[1] != b_shp_shape[0]:
                    findings.append({
                        "rule":     "R004",
                        "severity": "MEDIUM",
                        "cve":      "CVE-2021-29609",
                        "message":  (
                            f"서브그래프[{sg_idx}] SparseAdd 연산자[{op_idx}] "
                            f"b_indices.shape[1]={b_idx_shape[1]} != "
                            f"b_shape.shape[0]={b_shp_shape[0]} (차원 불일치)"
                        ),
                    })
    return findings


def _check_r005(model: TFLiteModel):
    """
    R005 (HIGH) CVE-2021-29591
    While 연산자의 cond 서브그래프와 body 서브그래프 인덱스가 동일한 경우.
    While: builtin_code=82, builtin_options_type=119 (WhileOptions)
    WhileOptions.cond_subgraph_index(field 0) == WhileOptions.body_subgraph_index(field 1)
    """
    findings = []

    # While opcode 인덱스 수집
    while_opcodes = {
        i for i, oc in enumerate(model.operator_codes)
        if oc.builtin_code == BUILTIN_WHILE
    }
    if not while_opcodes:
        return findings

    for sg_idx, sg in enumerate(model.subgraphs):
        for op_idx, op in enumerate(sg.operators):
            if op.opcode_index not in while_opcodes:
                continue
            # _parse_operator에서 builtin_options (WhileOptions 테이블)를 파싱하여
            # while_cond_subgraph, while_body_subgraph에 저장함
            if (op.while_cond_subgraph is not None
                    and op.while_body_subgraph is not None
                    and op.while_cond_subgraph == op.while_body_subgraph):
                findings.append({
                    "rule":     "R005",
                    "severity": "HIGH",
                    "cve":      "CVE-2021-29591",
                    "message":  (
                        f"서브그래프[{sg_idx}] While 연산자[{op_idx}] "
                        f"cond_subgraph={op.while_cond_subgraph} == "
                        f"body_subgraph={op.while_body_subgraph} "
                        "(동일 서브그래프 참조 — 무한루프/OOB 위험)"
                    ),
                })
    return findings

# ──────────────────────────────────────────────
# 위험도 점수 계산
# ──────────────────────────────────────────────

def compute_risk_score(findings):
    """
    위험도 점수 R(M) 계산
    W(M) = 10*n_C + 5*n_H + 2*n_M + n_L
    p_s  = (w_s * n_s) / W(M)
    H(p) = -sum(p_s * log2(p_s))  for p_s > 0
    C(M) = 1 - H(p) / log2(4)     in [0, 1]
    R(M) = log2(1 + W(M) + C(M))
    W(M)=0이면 R(M)=0
    """
    counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for f in findings:
        sev = f.get("severity", "LOW")
        counts[sev] = counts.get(sev, 0) + 1

    n_C, n_H, n_M, n_L = (
        counts["CRITICAL"], counts["HIGH"],
        counts["MEDIUM"],   counts["LOW"]
    )
    W = 10 * n_C + 5 * n_H + 2 * n_M + n_L

    if W == 0:
        return 0.0, 0.0, 0.0

    # 엔트로피 기반 복잡도 C(M)
    probs = []
    for sev, w_s, n_s in [
        ("CRITICAL", 10, n_C),
        ("HIGH",      5, n_H),
        ("MEDIUM",    2, n_M),
        ("LOW",       1, n_L),
    ]:
        if n_s > 0:
            probs.append((w_s * n_s) / W)

    H = -sum(p * math.log2(p) for p in probs if p > 0)
    C = 1.0 - H / math.log2(4) if H > 0 else 1.0
    C = max(0.0, min(1.0, C))  # [0, 1] 클리핑

    R = math.log2(1 + W + C)
    return W, C, R

# ──────────────────────────────────────────────
# 분석 실행
# ──────────────────────────────────────────────

def analyze(data: bytes, filename: str = ""):
    """
    TFLite 바이너리를 분석하여 발견 사항과 위험도 점수를 반환한다.
    반환값: dict (model 통계, findings 목록, 점수)
    """
    model = parse_tflite(data)

    all_findings = []
    all_findings.extend(_check_r001(model))
    all_findings.extend(_check_r002(model))
    all_findings.extend(_check_r003(model))
    all_findings.extend(_check_r004(model))
    all_findings.extend(_check_r005(model))

    W, C, R = compute_risk_score(all_findings)

    # 모델 통계 집계
    total_tensors = sum(len(sg.tensors) for sg in model.subgraphs)
    total_ops     = sum(len(sg.operators) for sg in model.subgraphs)

    return {
        "filename":    filename,
        "version":     model.version,
        "subgraphs":   len(model.subgraphs),
        "tensors":     total_tensors,
        "operators":   total_ops,
        "buffers":     len(model.buffers),
        "findings":    all_findings,
        "W":           W,
        "C":           C,
        "R":           R,
        "file_size":   len(data),
    }

# ──────────────────────────────────────────────
# 출력 포맷터
# ──────────────────────────────────────────────

def _sev_label(sev):
    """컬러 적용된 심각도 라벨 반환"""
    color = SEVERITY_COLOR.get(sev, ANSI_WHITE)
    return f"{color}{ANSI_BOLD}[{sev}]{ANSI_RESET}"


def print_report(result: dict, use_color: bool = True):
    """분석 결과를 콘솔에 출력한다"""
    filename  = result["filename"]
    findings  = result["findings"]
    W, C, R   = result["W"], result["C"], result["R"]

    sep = "─" * 60
    if use_color:
        header = f"{ANSI_CYAN}{ANSI_BOLD}{'TFLite SAST 분석 결과':^60}{ANSI_RESET}"
    else:
        header = f"{'TFLite SAST 분석 결과':^60}"

    print(sep)
    print(header)
    print(sep)
    print(f"  파일     : {filename}")
    print(f"  크기     : {result['file_size']:,} bytes  ({result['file_size']/1024:.2f} KB)")
    print(f"  버전     : {result['version']}")
    print(f"  서브그래프: {result['subgraphs']}개")
    print(f"  텐서     : {result['tensors']}개")
    print(f"  연산자   : {result['operators']}개")
    print(sep)

    if findings:
        print(f"  발견 사항 ({len(findings)}건):")
        for f in findings:
            sev   = f["severity"]
            label = _sev_label(sev) if use_color else f"[{sev}]"
            print(f"  {label} {f['rule']} ({f['cve']})")
            print(f"    └ {f['message']}")
    else:
        ok = f"{ANSI_GREEN}취약점 없음{ANSI_RESET}" if use_color else "취약점 없음"
        print(f"  발견 사항: {ok}")

    print(sep)
    print(f"  W(M) = {W:.0f}   C(M) = {C:.4f}   R(M) = {R:.4f}")
    print(sep)


# ──────────────────────────────────────────────
# CLI 진입점
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="TFLite 모델 파일 정적 보안 분석기 (표준 라이브러리 전용)"
    )
    parser.add_argument(
        "tflite_file",
        nargs="?",
        help="분석할 .tflite 파일 경로"
    )
    parser.add_argument(
        "--json",
        metavar="OUTPUT",
        help="결과를 JSON 파일로 저장"
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="ANSI 컬러 출력 비활성화"
    )
    parser.add_argument(
        "--dir",
        metavar="DIRECTORY",
        help="디렉터리 내 모든 .tflite 파일 일괄 분석"
    )
    args = parser.parse_args()

    use_color = not args.no_color

    results = []

    if args.dir:
        # 디렉터리 일괄 분석
        dirpath = Path(args.dir)
        files = sorted(dirpath.glob("*.tflite"))
        if not files:
            print(f"[경고] {args.dir} 에서 .tflite 파일을 찾을 수 없습니다.")
            return
        for fpath in files:
            data = fpath.read_bytes()
            result = analyze(data, str(fpath))
            print_report(result, use_color)
            results.append(result)

    elif args.tflite_file:
        fpath = Path(args.tflite_file)
        if not fpath.exists():
            print(f"[오류] 파일을 찾을 수 없습니다: {fpath}")
            return
        data = fpath.read_bytes()
        result = analyze(data, str(fpath))
        print_report(result, use_color)
        results.append(result)

    else:
        parser.print_help()
        return

    # JSON 저장
    if args.json:
        out_path = Path(args.json)
        # findings 내 bytes 직렬화 처리
        def _serialize(obj):
            if isinstance(obj, bytes):
                return obj.hex()
            return obj

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=_serialize)
        print(f"\n결과 저장 완료: {out_path}")


if __name__ == "__main__":
    main()
