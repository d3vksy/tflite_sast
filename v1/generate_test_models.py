#!/usr/bin/env python3
"""
generate_test_models.py -- TFLite 테스트 모델 생성기
FlatBuffer 바이너리를 Python struct 모듈로 직접 빌드한다.
정상 모델 15개(B001-B015)와 취약점 주입 모델 8개(M001-M008) 생성.
"""

import struct
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# 역방향(reverse) FlatBuffer 빌더
#
# 동작 원리:
#   - self.buf 앞쪽(인덱스 0)에 데이터를 삽입(prepend)한다.
#   - 나중에 쓴 것일수록 파일 앞쪽에 위치한다.
#   - "offset_from_end(OFE)" = 현재 buf 크기 기준,
#     어떤 객체가 buf 끝에서 몇 바이트 앞에 있는지를 나타낸다.
#   - 파일 최종 크기가 F이면 절대 위치 = F - OFE
#   - OFE는 prepend 시 값이 변하지 않는다는 것이 핵심 불변식이다.
#
# UOffset 계산 (두 위치 p_slot, p_ref 의 OFE 기준):
#   slot_abs = F - p_slot,  ref_abs = F - p_ref
#   UOffset_value = ref_abs - slot_abs = p_slot - p_ref
#   (ref가 먼저 빌드되었으면 p_ref < p_slot 이므로 양수 보장)
#
# vtable soffset 계산:
#   table_abs = F - table_end
#   vtable_abs = F - vtable_end  (vtable 이 table 앞에 위치)
#   soffset_value = table_abs - vtable_abs = vtable_end - table_end  (양수)
#   파서: vtable_offset = table_offset - soffset_value = vtable_abs ✓
#
# vtable 필드 오프셋:
#   field_abs = F - field_end
#   table_abs = F - table_end
#   vt_off = field_abs - table_abs = table_end - field_end
#   파서: return table_offset + rel = table_abs + vt_off = field_abs ✓
# ─────────────────────────────────────────────────────────────

class FlatBufferBuilder:
    """역방향 FlatBuffer 빌더 (prepend 방식)"""

    def __init__(self):
        self.buf = bytearray()
        self._object_end   = 0   # start_table 시점의 OFE
        self._vtable_fields = {}  # field_id -> OFE when field was written

    # ── 기본 유틸리티 ──────────────────────────────────────────

    def size(self):
        """현재 버퍼 크기 = 가장 최근에 기록된 객체의 OFE"""
        return len(self.buf)

    def _pre(self, data: bytes):
        """bytes를 버퍼 앞에 삽입"""
        self.buf[0:0] = data

    def _align(self, n: int):
        """size()가 n의 배수가 되도록 패딩 삽입"""
        rem = self.size() % n
        if rem:
            self._pre(b'\x00' * (n - rem))

    # ── 문자열 ─────────────────────────────────────────────────

    def create_string(self, s: str) -> int:
        """FlatBuffer 문자열 생성 → OFE 반환.
        파일 레이아웃: [length(u32)][content][null][trailing_pad]
        prepend 시 한 번에 전체 블록을 삽입하여 length-content 사이에
        패딩이 끼어드는 버그를 방지한다.
        """
        enc = s.encode('utf-8')
        data = struct.pack('<I', len(enc)) + enc + b'\x00'
        rem = len(data) % 4
        if rem:
            data += b'\x00' * (4 - rem)   # trailing padding (4바이트 정렬)
        self._pre(data)
        return self.size()

    # ── 벡터 ──────────────────────────────────────────────────

    def create_vector_i32(self, values: list) -> int:
        """int32 배열 벡터 → OFE 반환"""
        self._align(4)
        # reversed: 마지막 원소를 먼저 prepend → 파일에서는 정순
        for v in reversed(values):
            self._pre(struct.pack('<i', v))
        self._pre(struct.pack('<I', len(values)))
        self._align(4)
        return self.size()

    def create_vector_u8(self, values: list) -> int:
        """uint8 배열 벡터 (Buffer.data 등) → OFE 반환
        파일 레이아웃: [count(u32)][data bytes...]
        count 앞에 정렬 패딩이 올 수 있으므로 count 기록 직후 OFE를 캡처해
        반환한다. (패딩 추가 후의 size()를 반환하면 UOffset이 패딩을 가리킴)
        """
        for v in reversed(values):
            self._pre(struct.pack('<B', v & 0xFF))
        self._pre(struct.pack('<I', len(values)))
        count_ofe = self.size()   # count 위치 (패딩 추가 전)
        self._align(4)            # 파일 내 count 4-byte 정렬 보장 (앞쪽 패딩)
        return count_ofe          # UOffset은 반드시 count를 가리켜야 함

    def create_vector_of_offsets(self, ofe_list: list) -> int:
        """
        테이블/오브젝트의 OFE 리스트로 UOffset 벡터 생성 → OFE 반환.
        ofe_list[0]이 파일에서 벡터의 첫 번째 원소가 되도록 reversed 처리.
        """
        self._align(4)
        for ref_end in reversed(ofe_list):
            # 이 슬롯의 OFE: size() + 4 (4바이트 쓰기 직후)
            slot_end = self.size() + 4
            uoff = slot_end - ref_end   # 항상 양수 (ref는 먼저 빌드됨)
            self._pre(struct.pack('<I', uoff))
        self._pre(struct.pack('<I', len(ofe_list)))
        self._align(4)
        return self.size()

    # ── 테이블 ─────────────────────────────────────────────────

    def start_table(self):
        """테이블 빌드 시작"""
        self._object_end    = self.size()
        self._vtable_fields = {}

    def add_field_u8(self, field_id: int, value: int, default: int = 0):
        """uint8 / bool 스칼라 필드 추가"""
        if value == default:
            return
        self._pre(struct.pack('<B', value & 0xFF))
        self._vtable_fields[field_id] = self.size()

    def add_field_i32(self, field_id: int, value: int, default: int = 0):
        """int32 스칼라 필드 추가"""
        if value == default:
            return
        self._align(4)
        self._pre(struct.pack('<i', value))
        self._vtable_fields[field_id] = self.size()

    def add_field_u32(self, field_id: int, value: int, default: int = 0):
        """uint32 스칼라 필드 추가"""
        if value == default:
            return
        self._align(4)
        self._pre(struct.pack('<I', value & 0xFFFFFFFF))
        self._vtable_fields[field_id] = self.size()

    def add_field_offset(self, field_id: int, ref_end: int):
        """
        UOffset 필드 추가.
        ref_end: 참조 대상 객체의 OFE (반드시 이미 빌드된 객체).
        """
        self._align(4)
        slot_end = self.size() + 4   # 이 슬롯의 OFE (쓰기 직후)
        uoff = slot_end - ref_end    # 양수 보장
        self._pre(struct.pack('<I', uoff))
        self._vtable_fields[field_id] = self.size()

    def end_table(self) -> int:
        """
        테이블 빌드 완료: soffset + vtable 기록.
        반환: 테이블 객체의 OFE.
        """
        # soffset placeholder (4바이트)
        self._align(4)
        self._pre(b'\x00\x00\x00\x00')
        table_end = self.size()   # 테이블 soffset 의 OFE

        # 객체 크기 = soffset ~ 마지막 필드 끝 (파일 방향)
        obj_size = table_end - self._object_end

        # vtable 크기 계산
        max_fid  = max(self._vtable_fields.keys()) if self._vtable_fields else -1
        n_fields = max_fid + 1
        vt_size  = 4 + n_fields * 2  # vtable_size(u16) + obj_size(u16) + field_off(u16)*N

        # vtable 바이트 빌드
        vt = bytearray(vt_size)
        struct.pack_into('<H', vt, 0, vt_size)
        struct.pack_into('<H', vt, 2, obj_size)
        for fid, field_end in self._vtable_fields.items():
            vt_off = table_end - field_end   # 테이블 시작에서 필드까지 오프셋
            struct.pack_into('<H', vt, 4 + fid * 2, vt_off)

        # vtable prepend
        self._align(2)
        self._pre(bytes(vt))
        vtable_end = self.size()

        # soffset 패치: soffset_value = vtable_end - table_end (양수)
        idx = self.size() - table_end   # 버퍼 내 soffset 의 인덱스
        struct.pack_into('<i', self.buf, idx, vtable_end - table_end)

        return table_end   # 테이블 객체의 OFE

    # ── 파일 완성 ──────────────────────────────────────────────

    def finish(self, root_end: int) -> bytes:
        """
        FlatBuffer 파일 완성.
        root_end: 루트 테이블 객체의 OFE.
        반환: 완성된 바이너리 (파일 식별자 "TFL3" 포함).

        주의: 파일 식별자("TFL3") 앞에 4-byte 정렬 패딩을 삽입하여
        최종 버퍼 크기를 4의 배수로 맞춘다. 이렇게 해야
        모든 UOffset 필드의 절대 위치가 4-byte 정렬을 유지한다.
        (절대위치 = total_size - OFE, 둘 다 4의 배수이면 차이도 4의 배수)
        """
        # 파일 식별자 삽입 전 4-byte 정렬 보장
        self._align(4)
        # 파일 식별자 삽입
        self._pre(b'TFL3')
        # 루트 UOffset: 파일 시작(offset=0)에서 루트 객체까지 절대 거리
        # 최종 크기 = size() + 4 (루트 UOffset 4바이트 추가 후)
        final_size = self.size() + 4
        root_abs   = final_size - root_end
        self._pre(struct.pack('<I', root_abs))
        return bytes(self.buf)


# ─────────────────────────────────────────────────────────────
# TFLite 구조 빌더 헬퍼 함수
# ─────────────────────────────────────────────────────────────

def build_buffer(b: FlatBufferBuilder, data: bytes = b'') -> int:
    """Buffer 테이블 생성 (field 0 = data: uint8[])"""
    b.start_table()
    if data:
        vec = b.create_vector_u8(list(data))
        b.add_field_offset(0, vec)
    return b.end_table()


def build_tensor(b: FlatBufferBuilder,
                 name: str,
                 shape: list,
                 tensor_type: int = 1,    # FLOAT32
                 buffer_idx: int = 1,
                 is_variable: bool = False,
                 shape_signature: list = None) -> int:
    """
    Tensor 테이블 생성.
    실제 TFLite schema.fbs 기준:
      0=name, 1=shape, 2=type, 3=buffer, 4=quantization,
      5=is_variable(bool), 6=sparsity, 7=shape_signature([int])
    shape_signature: 동적 차원(-1)을 표현하는 필드 (TFLite 정식 방법).
                     shape 필드의 -1은 CVE-2022-23558 취약점 트리거이지만,
                     shape_signature의 -1은 정상적인 동적 배치 차원 표현.
    """
    name_off  = b.create_string(name)
    shape_off = b.create_vector_i32(shape)
    ss_off = None
    if shape_signature is not None:
        ss_off = b.create_vector_i32(shape_signature)
    b.start_table()
    b.add_field_offset(0, name_off)
    b.add_field_offset(1, shape_off)
    b.add_field_i32(2, tensor_type, 0)
    b.add_field_u32(3, buffer_idx, 0)
    if is_variable:
        b.add_field_u8(5, 1, 0)          # field 5 = is_variable (실제 스키마)
    if ss_off is not None:
        b.add_field_offset(7, ss_off)    # field 7 = shape_signature (실제 스키마)
    return b.end_table()


def build_operator(b: FlatBufferBuilder,
                   opcode_index: int,
                   inputs: list,
                   outputs: list,
                   builtin_options_type: int = 0,
                   builtin_options_ofe: int = None,
                   custom_options: bytes = b'') -> int:
    """
    Operator 테이블 생성.
    실제 TFLite schema.fbs 기준:
      0=opcode_index, 1=inputs, 2=outputs,
      3=builtin_options_type(u8), 4=builtin_options(union table UOffset),
      5=custom_options([ubyte])
    builtin_options_ofe: build_while_options_table() 등으로 미리 빌드한 OFE.
    """
    inputs_off  = b.create_vector_i32(inputs)
    outputs_off = b.create_vector_i32(outputs)
    b.start_table()
    b.add_field_u32(0, opcode_index, 0)
    b.add_field_offset(1, inputs_off)
    b.add_field_offset(2, outputs_off)
    if builtin_options_type != 0 and builtin_options_ofe is not None:
        b.add_field_u8(3, builtin_options_type, 0)      # field 3: type tag (u8)
        b.add_field_offset(4, builtin_options_ofe)      # field 4: union table UOffset
    if custom_options:
        co_off = b.create_vector_u8(list(custom_options))
        b.add_field_offset(5, co_off)
    return b.end_table()


def build_while_options_table(b: FlatBufferBuilder,
                               cond_idx: int,
                               body_idx: int) -> int:
    """
    WhileOptions 테이블 생성.
    실제 TFLite schema.fbs 기준:
      field 0 = cond_subgraph_index (int32)
      field 1 = body_subgraph_index (int32)
    build_operator()의 builtin_options_ofe 인자로 전달한다.
    반드시 build_operator() 호출 전에 먼저 빌드해야 한다.
    """
    b.start_table()
    b.add_field_i32(0, cond_idx, 0)
    b.add_field_i32(1, body_idx, 0)
    return b.end_table()


def build_opcode(b: FlatBufferBuilder,
                 builtin_code: int = 0,
                 custom_code: str = '',
                 deprecated: int = 0) -> int:
    """
    OperatorCode 테이블 생성.
    schema: 0=deprecated_builtin_code(u8), 1=custom_code, 3=builtin_code(i32)
    """
    b.start_table()
    if deprecated != 0:
        b.add_field_u8(0, deprecated, 0)
    if custom_code:
        cc_off = b.create_string(custom_code)
        b.add_field_offset(1, cc_off)
    if builtin_code != 0:
        b.add_field_i32(3, builtin_code, 0)
    return b.end_table()


def assemble_model(b: FlatBufferBuilder,
                   opcode_ofe_list: list,
                   subgraph_ofe_list: list,
                   buffer_ofe_list: list,
                   version: int = 3) -> bytes:
    """
    Model 테이블 조립 후 FlatBuffer 바이너리 반환.
    schema: 0=version, 1=operator_codes, 2=subgraphs, 4=buffers
    """
    oc_vec  = b.create_vector_of_offsets(opcode_ofe_list)
    sg_vec  = b.create_vector_of_offsets(subgraph_ofe_list)
    buf_vec = b.create_vector_of_offsets(buffer_ofe_list)
    b.start_table()
    b.add_field_u32(0, version, 0)
    b.add_field_offset(1, oc_vec)
    b.add_field_offset(2, sg_vec)
    b.add_field_offset(4, buf_vec)
    root = b.end_table()
    return b.finish(root)


def build_subgraph(b: FlatBufferBuilder,
                   tensor_ofe_list: list,
                   op_ofe_list: list,
                   inputs: list,
                   outputs: list,
                   name: str = 'main') -> int:
    """
    SubGraph 테이블 생성.
    schema: 0=tensors, 1=inputs, 2=outputs, 3=operators, 4=name
    """
    nm_off = b.create_string(name)
    t_vec  = b.create_vector_of_offsets(tensor_ofe_list)
    op_vec = b.create_vector_of_offsets(op_ofe_list)
    in_vec = b.create_vector_i32(inputs)
    ou_vec = b.create_vector_i32(outputs)
    b.start_table()
    b.add_field_offset(0, t_vec)
    b.add_field_offset(1, in_vec)
    b.add_field_offset(2, ou_vec)
    b.add_field_offset(3, op_vec)
    b.add_field_offset(4, nm_off)
    return b.end_table()


# ─────────────────────────────────────────────────────────────
# 정상 모델 (B001-B010)
# ─────────────────────────────────────────────────────────────

def _make_normal_model(seed: int) -> bytes:
    """취약점 없는 정상 TFLite 모델. seed로 텐서/연산자 수를 변화."""
    b = FlatBufferBuilder()
    n_tensors = 8 + (seed % 5)
    n_ops     = 7 + (seed % 4)

    # 버퍼: 인덱스 0은 빈 버퍼, 나머지는 더미 데이터
    buf_offs = [build_buffer(b)]
    for i in range(1, n_tensors + 1):
        buf_offs.append(build_buffer(b, bytes([i % 256] * 4)))

    # ADD opcode
    oc_offs = [build_opcode(b, builtin_code=0, deprecated=0)]

    # 텐서 (모두 is_variable=False, buffer_idx >= 1)
    tensor_offs = []
    for i in range(n_tensors):
        d1 = 1 + (i * 3 + seed) % 64
        d2 = 1 + (i * 7 + seed) % 32
        tensor_offs.append(build_tensor(b, f't{i}', [1, d1, d2, 1],
                                        buffer_idx=i + 1))

    # 연산자 (출력 텐서도 buffer_idx >= 1)
    op_offs = []
    for i in range(n_ops):
        op_offs.append(build_operator(b, 0,
                                      inputs=[i % n_tensors,
                                              (i + 1) % n_tensors],
                                      outputs=[(i + 2) % n_tensors]))

    sg = build_subgraph(b, tensor_offs, op_offs,
                        inputs=[0], outputs=[n_tensors - 1])
    return assemble_model(b, oc_offs, [sg], buf_offs)


# ─────────────────────────────────────────────────────────────
# 취약점 주입 모델
# ─────────────────────────────────────────────────────────────

def make_m001_r001_large_shape() -> bytes:
    """M001: R001 -- shape에 2^31-1 삽입 (int32 최댓값, 2^30 초과) (CVE-2022-23558)"""
    b = FlatBufferBuilder()
    buf_offs = [build_buffer(b), build_buffer(b, b'\x01\x02\x03\x04')]
    oc_offs  = [build_opcode(b, 0, deprecated=0)]

    LARGE = (1 << 31) - 1   # 2147483647 (int32 최댓값, > 2^30)
    t0 = build_tensor(b, 'normal',      [1, 224, 224, 3], buffer_idx=1)
    t1 = build_tensor(b, 'vuln_tensor', [1, LARGE, 4, 1], buffer_idx=1)
    t2 = build_tensor(b, 'output',      [1, 4, 4, 1],     buffer_idx=1)
    op = build_operator(b, 0, [0, 1], [2])
    sg = build_subgraph(b, [t0, t1, t2], [op], [0], [2])
    return assemble_model(b, oc_offs, [sg], buf_offs)


def make_m002_r002_variable_input() -> bytes:
    """M002: R002 -- is_variable=True 텐서를 연산자 입력으로 사용 (CVE-2021-37681)"""
    b = FlatBufferBuilder()
    buf_offs = [build_buffer(b), build_buffer(b, b'\xDE\xAD\xBE\xEF')]
    oc_offs  = [build_opcode(b, 0, deprecated=0)]

    t0 = build_tensor(b, 'normal_in',   [1, 8, 8, 1], buffer_idx=1, is_variable=False)
    t1 = build_tensor(b, 'variable_in', [1, 8, 8, 1], buffer_idx=1, is_variable=True)
    t2 = build_tensor(b, 'output',      [1, 8, 8, 1], buffer_idx=1, is_variable=False)
    op = build_operator(b, 0, [0, 1], [2])
    sg = build_subgraph(b, [t0, t1, t2], [op], [0], [2])
    return assemble_model(b, oc_offs, [sg], buf_offs)


def make_m003_r003_splitv_oob() -> bytes:
    """M003: R003 -- SPLIT_V axis=5, input rank=2 → axis OOB (CVE-2021-29606)"""
    import struct as _struct
    b = FlatBufferBuilder()
    # buffer[0]: 빈(관례), buffer[1]: input [4,4] float32, buffer[2]: size_splits=[2,2], buffer[3]: axis=5
    buf_offs = [
        build_buffer(b),
        build_buffer(b, bytes(64)),                         # 4*4*4 = 64 bytes float32
        build_buffer(b, _struct.pack('<2i', 2, 2)),         # size_splits
        build_buffer(b, _struct.pack('<i', 5)),             # axis=5 (rank=2 초과 → OOB)
        build_buffer(b),                                    # output0 (빈)
        build_buffer(b),                                    # output1 (빈)
    ]
    oc_offs = [build_opcode(b, builtin_code=102, deprecated=127)]  # SPLIT_V

    t0 = build_tensor(b, 'input',       [4, 4], tensor_type=1, buffer_idx=1)   # FLOAT32
    t1 = build_tensor(b, 'size_splits', [2],    tensor_type=2, buffer_idx=2)   # INT32
    t2 = build_tensor(b, 'axis',        [],     tensor_type=2, buffer_idx=3)   # INT32 scalar
    t3 = build_tensor(b, 'output0',     [2, 4], tensor_type=1, buffer_idx=4)
    t4 = build_tensor(b, 'output1',     [2, 4], tensor_type=1, buffer_idx=5)

    op = build_operator(b, 0, [0, 1, 2], [3, 4])
    sg = build_subgraph(b, [t0, t1, t2, t3, t4], [op], [0], [3, 4])
    return assemble_model(b, oc_offs, [sg], buf_offs)


def make_m004_r004_sparseadd_mismatch() -> bytes:
    """M004: R004 -- SparseAdd indices/shape 불일치 (CVE-2021-29609)"""
    b = FlatBufferBuilder()
    buf_offs = [build_buffer(b)] + [build_buffer(b, b'\x01') for _ in range(8)]
    oc_offs = [build_opcode(b, builtin_code=0, custom_code='SparseAdd', deprecated=127)]

    t0 = build_tensor(b, 'a_values',  [5],    buffer_idx=1)
    t1 = build_tensor(b, 'a_indices', [5, 3], buffer_idx=2)
    t2 = build_tensor(b, 'a_shape',   [4],    buffer_idx=3)  # 3 != 4 → R004
    t3 = build_tensor(b, 'b_values',  [3],    buffer_idx=4)
    t4 = build_tensor(b, 'b_indices', [3, 2], buffer_idx=5)
    t5 = build_tensor(b, 'b_shape',   [2],    buffer_idx=6)
    t6 = build_tensor(b, 'output',    [8],    buffer_idx=7)

    op = build_operator(b, 0, [0, 1, 2, 3, 4, 5], [6])
    sg = build_subgraph(b, [t0, t1, t2, t3, t4, t5, t6], [op], [0], [6])
    return assemble_model(b, oc_offs, [sg], buf_offs)


def make_m005_r005_while_same_subgraph() -> bytes:
    """M005: R005 -- While body=cond 동일 인덱스 (CVE-2021-29591)"""
    b = FlatBufferBuilder()
    buf_offs = [build_buffer(b), build_buffer(b, b'\x01\x02\x03\x04')]
    oc_offs = [build_opcode(b, builtin_code=82, deprecated=127)]

    while_opts = build_while_options_table(b, cond_idx=1, body_idx=1)
    t0 = build_tensor(b, 'loop_var', [1], buffer_idx=1)
    t1 = build_tensor(b, 'output',   [1], buffer_idx=1)
    op = build_operator(b, 0, [0], [1],
                        builtin_options_type=119,
                        builtin_options_ofe=while_opts)
    sg = build_subgraph(b, [t0, t1], [op], [0], [1])
    return assemble_model(b, oc_offs, [sg], buf_offs)


def make_m006_r005_while_same_subgraph() -> bytes:
    """M006: R005 -- While body=cond 동일 인덱스 (CVE-2021-29591)
    builtin_options 필드에 WhileOptions 테이블을 올바르게 저장한다.
    """
    b = FlatBufferBuilder()
    buf_offs = [build_buffer(b), build_buffer(b, b'\x01\x02\x03\x04')]
    # While opcode (builtin_code=82)
    oc_offs = [build_opcode(b, builtin_code=82, deprecated=127)]

    # WhileOptions: cond=1, body=1 (동일 → R005 탐지)
    while_opts = build_while_options_table(b, cond_idx=1, body_idx=1)
    t0 = build_tensor(b, 'loop_var', [1], buffer_idx=1)
    t1 = build_tensor(b, 'output',   [1], buffer_idx=1)
    op = build_operator(b, 0, [0], [1],
                        builtin_options_type=119,
                        builtin_options_ofe=while_opts)
    sg = build_subgraph(b, [t0, t1], [op], [0], [1])
    return assemble_model(b, oc_offs, [sg], buf_offs)


def make_m007_r001_r002_combo() -> bytes:
    """M007: R001+R002 복합 패턴"""
    b = FlatBufferBuilder()
    buf_offs = [build_buffer(b), build_buffer(b, b'\x01\x02\x03\x04')]
    oc_offs  = [build_opcode(b, 0, deprecated=0)]

    LARGE = (1 << 31) - 1   # int32 최댓값 > 2^30
    t0 = build_tensor(b, 'large_shape', [1, LARGE, 4, 1], buffer_idx=1, is_variable=False)
    t1 = build_tensor(b, 'variable_in', [1,    4, 4, 1], buffer_idx=1, is_variable=True)
    t2 = build_tensor(b, 'output',      [1,    4, 4, 1], buffer_idx=1, is_variable=False)
    op = build_operator(b, 0, [0, 1], [2])
    sg = build_subgraph(b, [t0, t1, t2], [op], [0], [2])
    return assemble_model(b, oc_offs, [sg], buf_offs)


def make_m008_r003_r004_r005_combo() -> bytes:
    """M008: R003(SPLIT_V axis OOB)+R004(SparseAdd)+R005(While) 복합 패턴"""
    import struct as _struct
    b = FlatBufferBuilder()

    # R003용 버퍼: input[4,4], size_splits=[2,2], axis=5 (OOB)
    buf_offs = [
        build_buffer(b),                                      # buffer[0]: 빈(관례)
        build_buffer(b, bytes(64)),                           # buffer[1]: SPLIT_V input [4,4]
        build_buffer(b, _struct.pack('<2i', 2, 2)),           # buffer[2]: size_splits
        build_buffer(b, _struct.pack('<i', 5)),               # buffer[3]: axis=5 (OOB)
        build_buffer(b),                                      # buffer[4]: SPLIT_V out0
        build_buffer(b),                                      # buffer[5]: SPLIT_V out1
        build_buffer(b, b'\x01'),                             # buffer[6]: SparseAdd a_values
        build_buffer(b, b'\x01'),                             # buffer[7]: SparseAdd a_indices
        build_buffer(b, b'\x01'),                             # buffer[8]: SparseAdd a_shape
        build_buffer(b, b'\x01'),                             # buffer[9]: SparseAdd b_values
        build_buffer(b, b'\x01'),                             # buffer[10]: SparseAdd b_indices
        build_buffer(b, b'\x01'),                             # buffer[11]: SparseAdd b_shape
        build_buffer(b, b'\x01'),                             # buffer[12]: SparseAdd output
        build_buffer(b, b'\x01\x02\x03\x04'),                # buffer[13]: While tensors
    ]

    oc0 = build_opcode(b, builtin_code=102, deprecated=127)              # SPLIT_V
    oc1 = build_opcode(b, builtin_code=0, custom_code='SparseAdd', deprecated=127)
    oc2 = build_opcode(b, builtin_code=82, deprecated=127)               # While
    oc_offs = [oc0, oc1, oc2]

    while_opts = build_while_options_table(b, cond_idx=1, body_idx=1)   # R005

    # R003: SPLIT_V axis=5, input rank=2
    t0  = build_tensor(b, 'sv_input',   [4, 4], tensor_type=1, buffer_idx=1)
    t1  = build_tensor(b, 'sv_splits',  [2],    tensor_type=2, buffer_idx=2)
    t2  = build_tensor(b, 'sv_axis',    [],     tensor_type=2, buffer_idx=3)  # axis=5
    t3  = build_tensor(b, 'sv_out0',    [2, 4], tensor_type=1, buffer_idx=4)
    t4  = build_tensor(b, 'sv_out1',    [2, 4], tensor_type=1, buffer_idx=5)
    # R004: SparseAdd indices/shape 불일치
    t5  = build_tensor(b, 'sa_aval',    [5],    buffer_idx=6)
    t6  = build_tensor(b, 'sa_aidx',    [5, 3], buffer_idx=7)
    t7  = build_tensor(b, 'sa_ashp',    [4],    buffer_idx=8)   # 3 != 4 → R004
    t8  = build_tensor(b, 'sa_bval',    [3],    buffer_idx=9)
    t9  = build_tensor(b, 'sa_bidx',    [3, 2], buffer_idx=10)
    t10 = build_tensor(b, 'sa_bshp',    [2],    buffer_idx=11)
    t11 = build_tensor(b, 'sa_out',     [8],    buffer_idx=12)
    # R005: While 동일 서브그래프
    t12 = build_tensor(b, 'while_in',   [1],    buffer_idx=13)
    t13 = build_tensor(b, 'while_out',  [1],    buffer_idx=13)

    op0 = build_operator(b, 0, [0, 1, 2], [3, 4])           # SPLIT_V
    op1 = build_operator(b, 1, [5, 6, 7, 8, 9, 10], [11])   # SparseAdd
    op2 = build_operator(b, 2, [12], [13],                   # While
                         builtin_options_type=119,
                         builtin_options_ofe=while_opts)

    sg = build_subgraph(b,
                        [t0,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13],
                        [op0, op1, op2],
                        [0], [13])
    return assemble_model(b, oc_offs, [sg], buf_offs)


# ─────────────────────────────────────────────────────────────
# "유사 패턴" 정상 모델 (B011-B015) — FPR 측정용
#
# 각 모델은 취약점 규칙의 조건과 "비슷해 보이지만"
# 실제로는 안전한 구조를 가진다.
# 분석기가 이 모델에서 탐지를 발생시키면 FP(오탐)이다.
# ─────────────────────────────────────────────────────────────

def make_b011_dynamic_shape_minus1() -> bytes:
    """
    B011: R001 유사 패턴 — 정상적인 동적 shape (-1 사용)
    TFLite의 동적 배치 차원은 shape_signature 필드(field 8)에 -1로 표현한다.
    shape 필드(field 1)는 정적 값(-1 없음)을 가져야 한다.

    핵심 구분:
      shape           = [1, 224, 224, 3]  ← 정적 (R001 탐지 안 함)
      shape_signature = [-1, 224, 224, 3] ← 동적 배치 차원 (TFLite 정식 방법)

    R001 조건 (dim < 0 in shape)에 해당하지 않으므로 탐지가 발생해서는 안 된다.
    """
    b = FlatBufferBuilder()
    buf_offs = [build_buffer(b), build_buffer(b, b'\x01\x02\x03\x04')]
    oc_offs  = [build_opcode(b, 0, deprecated=0)]

    # shape: 정적 값 (음수 없음) / shape_signature: 동적 배치 차원 -1
    t0 = build_tensor(b, 'dynamic_input',  [1, 224, 224, 3], buffer_idx=1,
                      shape_signature=[-1, 224, 224, 3])
    t1 = build_tensor(b, 'static_weights', [1,   3,   3, 3], buffer_idx=1)
    t2 = build_tensor(b, 'output',         [1, 224, 224, 1], buffer_idx=1,
                      shape_signature=[-1, 224, 224, 1])
    op = build_operator(b, 0, [0, 1], [2])
    sg = build_subgraph(b, [t0, t1, t2], [op], [0], [2])
    return assemble_model(b, oc_offs, [sg], buf_offs)


def make_b012_variable_tensor_not_input() -> bytes:
    """
    B012: R002 유사 패턴 — is_variable=True 텐서가 존재하지만
    연산자 입력으로 참조되지 않고 서브그래프 output만 사용
    (LSTM/GRU 상태 텐서 등 정상적인 가변 텐서 사용 패턴)
    """
    b = FlatBufferBuilder()
    buf_offs = [build_buffer(b), build_buffer(b, b'\x01\x02\x03\x04')]
    oc_offs  = [build_opcode(b, 0, deprecated=0)]

    t0 = build_tensor(b, 'input',          [1, 8, 8, 1], buffer_idx=1, is_variable=False)
    t1 = build_tensor(b, 'weights',        [1, 8, 8, 1], buffer_idx=1, is_variable=False)
    t2 = build_tensor(b, 'output',         [1, 8, 8, 1], buffer_idx=1, is_variable=False)
    # is_variable=True 이지만 연산자 입력으로 참조 안 됨 → R002 비탐지여야 함
    t3 = build_tensor(b, 'state_variable', [1, 4],       buffer_idx=1, is_variable=True)

    op = build_operator(b, 0, [0, 1], [2])   # t3은 입력/출력에 없음
    sg = build_subgraph(b, [t0, t1, t2, t3], [op], [0], [2])
    return assemble_model(b, oc_offs, [sg], buf_offs)


def make_b013_splitv_valid_axis() -> bytes:
    """
    B013: R003 유사 패턴 — SPLIT_V 연산자이지만 axis=1 (rank=2 이내, 정상)
    axis가 [0, rank) 범위 내이므로 R003이 탐지되어서는 안 된다.
    """
    import struct as _struct
    b = FlatBufferBuilder()
    buf_offs = [
        build_buffer(b),
        build_buffer(b, bytes(64)),                   # input [4,4] float32
        build_buffer(b, _struct.pack('<2i', 2, 2)),   # size_splits=[2,2]
        build_buffer(b, _struct.pack('<i', 1)),       # axis=1 (정상: rank=2 이내)
        build_buffer(b),                              # output0
        build_buffer(b),                              # output1
    ]
    oc_offs = [build_opcode(b, builtin_code=102, deprecated=127)]  # SPLIT_V

    t0 = build_tensor(b, 'input',       [4, 4], tensor_type=1, buffer_idx=1)
    t1 = build_tensor(b, 'size_splits', [2],    tensor_type=2, buffer_idx=2)
    t2 = build_tensor(b, 'axis',        [],     tensor_type=2, buffer_idx=3)  # axis=1 → 정상
    t3 = build_tensor(b, 'output0',     [4, 2], tensor_type=1, buffer_idx=4)
    t4 = build_tensor(b, 'output1',     [4, 2], tensor_type=1, buffer_idx=5)

    op = build_operator(b, 0, [0, 1, 2], [3, 4])
    sg = build_subgraph(b, [t0, t1, t2, t3, t4], [op], [0], [3, 4])
    return assemble_model(b, oc_offs, [sg], buf_offs)


def make_b014_sparseadd_matching() -> bytes:
    """
    B014: R004 유사 패턴 — SparseAdd opcode가 있지만
    indices/shape 차원이 일치하는 정상 케이스
    """
    b = FlatBufferBuilder()
    buf_offs = [build_buffer(b)] + [build_buffer(b, b'\x01') for _ in range(7)]
    oc_offs = [build_opcode(b, builtin_code=0, custom_code='SparseAdd', deprecated=127)]

    # a_indices.shape = [5, 3] → dim[1]=3
    # a_shape.shape   = [3]   → size=3  ← 일치! → R004 비탐지여야 함
    t0 = build_tensor(b, 'a_values',  [5],    buffer_idx=1)
    t1 = build_tensor(b, 'a_indices', [5, 3], buffer_idx=2)
    t2 = build_tensor(b, 'a_shape',   [3],    buffer_idx=3)  # 3 == 3 ✓
    t3 = build_tensor(b, 'b_values',  [3],    buffer_idx=4)
    t4 = build_tensor(b, 'b_indices', [3, 2], buffer_idx=5)
    t5 = build_tensor(b, 'b_shape',   [2],    buffer_idx=6)  # 2 == 2 ✓
    t6 = build_tensor(b, 'output',    [8],    buffer_idx=7)
    op = build_operator(b, 0, [0, 1, 2, 3, 4, 5], [6])
    sg = build_subgraph(b, [t0, t1, t2, t3, t4, t5, t6], [op], [0], [6])
    return assemble_model(b, oc_offs, [sg], buf_offs)


def make_b015_while_different_subgraph() -> bytes:
    """
    B015: R005 유사 패턴 — While opcode가 있지만
    cond/body 서브그래프 인덱스가 다른 정상 케이스.
    builtin_options 필드에 WhileOptions 테이블을 올바르게 저장한다.
    """
    b = FlatBufferBuilder()
    buf_offs = [build_buffer(b), build_buffer(b, b'\x01\x02\x03\x04')]
    oc_offs = [build_opcode(b, builtin_code=82, deprecated=127)]

    # WhileOptions: cond=0, body=1 → 서로 다름 → R005 비탐지여야 함
    while_opts = build_while_options_table(b, cond_idx=0, body_idx=1)
    t0 = build_tensor(b, 'loop_var', [1], buffer_idx=1)
    t1 = build_tensor(b, 'output',   [1], buffer_idx=1)
    op = build_operator(b, 0, [0], [1],
                        builtin_options_type=119,
                        builtin_options_ofe=while_opts)
    sg = build_subgraph(b, [t0, t1], [op], [0], [1])
    return assemble_model(b, oc_offs, [sg], buf_offs)


# ─────────────────────────────────────────────────────────────
# 모델 명세 및 생성 실행
# ─────────────────────────────────────────────────────────────

MODEL_SPECS = [
    ('B001.tflite', '정상 모델 #1',                        lambda: _make_normal_model(1)),
    ('B002.tflite', '정상 모델 #2',                        lambda: _make_normal_model(2)),
    ('B003.tflite', '정상 모델 #3',                        lambda: _make_normal_model(3)),
    ('B004.tflite', '정상 모델 #4',                        lambda: _make_normal_model(4)),
    ('B005.tflite', '정상 모델 #5',                        lambda: _make_normal_model(5)),
    ('B006.tflite', '정상 모델 #6',                        lambda: _make_normal_model(6)),
    ('B007.tflite', '정상 모델 #7',                        lambda: _make_normal_model(7)),
    ('B008.tflite', '정상 모델 #8',                        lambda: _make_normal_model(8)),
    ('B009.tflite', '정상 모델 #9',                        lambda: _make_normal_model(9)),
    ('B010.tflite', '정상 모델 #10',                       lambda: _make_normal_model(10)),
    # FPR 측정용 "유사 패턴" 정상 모델 (B011-B015)
    ('B011.tflite', 'FPR용: 동적 shape(-1) 정상 사용',    make_b011_dynamic_shape_minus1),
    ('B012.tflite', 'FPR용: is_variable 텐서 미참조',      make_b012_variable_tensor_not_input),
    ('B013.tflite', 'FPR용: SPLIT_V axis 정상(범위 내)',    make_b013_splitv_valid_axis),
    ('B014.tflite', 'FPR용: SparseAdd 정상(일치)',         make_b014_sparseadd_matching),
    ('B015.tflite', 'FPR용: While 다른 서브그래프 인덱스', make_b015_while_different_subgraph),
    ('M001.tflite', 'R001: 대형 shape (2^31)',             make_m001_r001_large_shape),
    ('M002.tflite', 'R002: is_variable 텐서 입력 참조',    make_m002_r002_variable_input),
    ('M003.tflite', 'R003: SPLIT_V axis OOB',              make_m003_r003_splitv_oob),
    ('M004.tflite', 'R004: SparseAdd indices/shape 불일치', make_m004_r004_sparseadd_mismatch),
    ('M005.tflite', 'R005: While body=cond 동일 (v1)',     make_m005_r005_while_same_subgraph),
    ('M006.tflite', 'R005: While body=cond 동일 (v2)',     make_m006_r005_while_same_subgraph),
    ('M007.tflite', 'R001+R002 복합',                      make_m007_r001_r002_combo),
    ('M008.tflite', 'R003+R004+R005 복합',                 make_m008_r003_r004_r005_combo),
]


def generate_all(output_dir: str = 'test_models') -> Path:
    out = Path(output_dir)
    out.mkdir(exist_ok=True)
    print(f'[*] 테스트 모델 생성 시작 -> {out.resolve()}')
    for filename, desc, fn in MODEL_SPECS:
        fpath = out / filename
        try:
            data = fn()
            fpath.write_bytes(data)
            print(f'  [OK] {filename:16s}  {len(data):6,} bytes  -- {desc}')
        except Exception as e:
            import traceback
            print(f'  [FAIL] {filename}: {e}')
            traceback.print_exc()
    print(f'[*] 완료: {len(MODEL_SPECS)}개 모델 생성')
    return out


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='TFLite 테스트 모델 생성기')
    parser.add_argument('--output-dir', '-o', default='test_models')
    args = parser.parse_args()
    generate_all(args.output_dir)
