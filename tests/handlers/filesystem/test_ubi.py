import struct
import zlib

import pytest

from unblob.handlers.filesystem.ubi import UBIFSHandler, UBIHandler
from unblob.testing import calculate_chunk, find_chunks

PEB_SIZE = 1024
VID_HDR_OFFSET = 64
DATA_OFFSET = 128
IMAGE_SEQ = 0xDEADBEEF
OTHER_IMAGE_SEQ = 0xC0FFEE11


def ec_header(image_seq=IMAGE_SEQ, *, corrupt_crc=False):
    header = struct.pack(
        ">4sB3sQIII32s",
        b"UBI#",
        1,
        b"\0" * 3,
        0,
        VID_HDR_OFFSET,
        DATA_OFFSET,
        image_seq,
        b"\0" * 32,
    )
    crc = zlib.crc32(header) ^ 0xFFFFFFFF
    if corrupt_crc:
        crc ^= 0xFF
    return header + struct.pack(">I", crc)


def peb(image_seq=IMAGE_SEQ, *, corrupt_crc=False):
    header = ec_header(image_seq=image_seq, corrupt_crc=corrupt_crc)
    return header + b"\xff" * (PEB_SIZE - len(header))


def bad_peb():
    # a bad eraseblock: garbage with a UBI# magic but no valid EC header CRC
    header = ec_header(corrupt_crc=True)
    return header + b"\xee" * (PEB_SIZE - len(header))


def ubi_image(*pebs):
    return b"".join(pebs)


class TestUBIHandler:
    def test_calculate_chunk(self):
        content = ubi_image(*(peb() for _ in range(4)))
        chunk = calculate_chunk(UBIHandler, content)
        assert (chunk.start_offset, chunk.end_offset) == (0, 4 * PEB_SIZE)

    def test_calculate_chunk_embedded(self):
        prefix = b"PREFIX"
        content = prefix + ubi_image(*(peb() for _ in range(3)))
        chunk = calculate_chunk(UBIHandler, content, start_offset=len(prefix))
        assert (chunk.start_offset, chunk.end_offset) == (
            len(prefix),
            len(prefix) + 3 * PEB_SIZE,
        )

    def test_calculate_chunk_skips_bad_blocks(self):
        content = ubi_image(peb(), peb(), bad_peb(), peb(), bad_peb(), peb())
        chunk = calculate_chunk(UBIHandler, content)
        assert (chunk.start_offset, chunk.end_offset) == (0, 6 * PEB_SIZE)

    def test_calculate_chunk_stops_after_too_many_bad_blocks(self):
        bad_peb_count = UBIHandler._MAX_CONSECUTIVE_BAD_PEBS + 1  # noqa: SLF001
        content = ubi_image(
            peb(), peb(), *(bad_peb() for _ in range(bad_peb_count)), peb()
        )
        chunk = calculate_chunk(UBIHandler, content)
        assert (chunk.start_offset, chunk.end_offset) == (0, 2 * PEB_SIZE)

    def test_calculate_chunk_bad_blocks_at_end_of_image(self):
        content = ubi_image(peb(), peb(), bad_peb(), bad_peb())
        chunk = calculate_chunk(UBIHandler, content)
        assert (chunk.start_offset, chunk.end_offset) == (0, 2 * PEB_SIZE)

    def test_calculate_chunk_truncated_peb_is_not_included(self):
        content = ubi_image(peb(), peb(), peb())[: 2 * PEB_SIZE + 512]
        chunk = calculate_chunk(UBIHandler, content)
        assert (chunk.start_offset, chunk.end_offset) == (0, 2 * PEB_SIZE)

    def test_calculate_chunk_first_peb_with_corrupt_crc(self):
        content = ubi_image(peb(corrupt_crc=True), peb(), peb())
        chunk = calculate_chunk(UBIHandler, content)
        assert (chunk.start_offset, chunk.end_offset) == (0, 3 * PEB_SIZE)

    def test_calculate_chunk_multiple_images_with_different_peb_sizes(self):
        image_a = ubi_image(*(peb(image_seq=IMAGE_SEQ) for _ in range(3)))
        other_peb_size = 2 * PEB_SIZE
        other_peb = ec_header(image_seq=OTHER_IMAGE_SEQ) + b"\xff" * (
            other_peb_size - 64
        )
        image_b = b"".join(other_peb for _ in range(4))
        content = image_a + image_b

        chunk_a = calculate_chunk(UBIHandler, content, start_offset=0)
        assert (chunk_a.start_offset, chunk_a.end_offset) == (0, len(image_a))

        chunk_b = calculate_chunk(UBIHandler, content, start_offset=len(image_a))
        assert (chunk_b.start_offset, chunk_b.end_offset) == (
            len(image_a),
            len(image_a) + 4 * other_peb_size,
        )

    def test_calculate_chunk_single_peb_is_rejected(self):
        content = ubi_image(peb())
        assert find_chunks(UBIHandler, content) == []

    def test_calculate_chunk_no_valid_ec_header_is_rejected(self):
        content = ubi_image(peb(corrupt_crc=True), peb(corrupt_crc=True))
        assert find_chunks(UBIHandler, content) == []


SB_NODE_SIZE = 4096
LEB_SIZE = 64
LEB_CNT = 4
MIN_IO_SIZE = 8


def ubifs_sb_node(
    endian="<",
    node_len=SB_NODE_SIZE,
    key_hash=0,
    key_fmt=0,
    min_io_size=MIN_IO_SIZE,
    leb_size=LEB_SIZE,
    leb_cnt=LEB_CNT,
    max_leb_cnt=LEB_CNT,
    jhead_cnt=1,
    fmt_version=4,
    default_compr=2,
):
    node = bytearray(SB_NODE_SIZE)
    struct.pack_into(f"{endian}I", node, 0, 0x06101831)  # magic
    struct.pack_into(f"{endian}I", node, 16, node_len)  # ch.len
    struct.pack_into(f"{endian}BB", node, 20, 6, 0)  # node_type, group_type
    struct.pack_into(f"{endian}BB", node, 26, key_hash, key_fmt)
    struct.pack_into(f"{endian}I", node, 32, min_io_size)
    struct.pack_into(f"{endian}I", node, 36, leb_size)
    struct.pack_into(f"{endian}I", node, 40, leb_cnt)
    struct.pack_into(f"{endian}I", node, 44, max_leb_cnt)
    struct.pack_into(f"{endian}I", node, 68, jhead_cnt)
    struct.pack_into(f"{endian}I", node, 80, fmt_version)
    struct.pack_into(f"{endian}H", node, 84, default_compr)
    return bytes(node)


def ubifs_image(**sb_kwargs):
    sb_node = ubifs_sb_node(**sb_kwargs)
    leb_size = sb_kwargs.get("leb_size", LEB_SIZE)
    leb_cnt = sb_kwargs.get("leb_cnt", LEB_CNT)
    return sb_node + b"\xff" * (leb_size * leb_cnt - len(sb_node))


class TestUBIFSHandler:
    def test_calculate_chunk(self):
        content = ubifs_image()
        chunk = calculate_chunk(UBIFSHandler, content)
        assert (chunk.start_offset, chunk.end_offset) == (0, LEB_SIZE * LEB_CNT)

    def test_calculate_chunk_big_endian(self):
        content = ubifs_image(endian=">")
        chunk = calculate_chunk(UBIFSHandler, content)
        assert (chunk.start_offset, chunk.end_offset) == (0, LEB_SIZE * LEB_CNT)

    def test_calculate_chunk_embedded(self):
        prefix = b"PREFIX"
        content = prefix + ubifs_image()
        chunk = calculate_chunk(UBIFSHandler, content, start_offset=len(prefix))
        assert (chunk.start_offset, chunk.end_offset) == (
            len(prefix),
            len(prefix) + LEB_SIZE * LEB_CNT,
        )

    def test_calculate_chunk_truncated(self):
        content = ubifs_image()[: LEB_SIZE * LEB_CNT - 1]
        assert find_chunks(UBIFSHandler, content) == []

    @pytest.mark.parametrize(
        "sb_kwargs",
        [
            pytest.param({"node_len": 100}, id="node_len"),
            pytest.param({"key_hash": 2}, id="key_hash"),
            pytest.param({"key_fmt": 2}, id="key_fmt"),
            pytest.param({"default_compr": 4}, id="default_compr"),
            pytest.param({"fmt_version": 0}, id="fmt_version_zero"),
            pytest.param({"fmt_version": 6}, id="fmt_version_future"),
            pytest.param({"jhead_cnt": 0}, id="jhead_cnt"),
            pytest.param({"min_io_size": 0}, id="min_io_size_zero"),
            pytest.param({"min_io_size": 7}, id="min_io_size_not_power_of_two"),
            pytest.param({"leb_size": 0}, id="leb_size_zero"),
            pytest.param({"leb_size": 63}, id="leb_size_misaligned"),
            pytest.param({"leb_cnt": 0}, id="leb_cnt_zero"),
            pytest.param(
                {"leb_cnt": 2, "max_leb_cnt": 1}, id="leb_cnt_above_max_leb_cnt"
            ),
        ],
    )
    def test_calculate_chunk_invalid_superblock_is_rejected(self, sb_kwargs):
        content = ubifs_image(**sb_kwargs)
        assert find_chunks(UBIFSHandler, content) == []
