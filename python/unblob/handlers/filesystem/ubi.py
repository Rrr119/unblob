import math
import shutil
import zlib
from enum import IntEnum
from pathlib import Path

from structlog import get_logger

from unblob.extractors import Command

from ...file_utils import (
    Endian,
    InvalidInputFormat,
    SeekError,
    get_endian,
    iterate_patterns,
)
from ...iter_utils import get_intervals
from ...models import (
    File,
    HandlerDoc,
    HandlerType,
    HexString,
    Reference,
    StructHandler,
    ValidChunk,
)

logger = get_logger()


# See the UBIFS node definitions in the Linux kernel:
# https://github.com/torvalds/linux/blob/v6.12/fs/ubifs/ubifs-media.h
class UBIFSKeyHash(IntEnum):
    R5 = 0
    TEST = 1


class UBIFSKeyFormat(IntEnum):
    SIMPLE = 0
    HASHED = 1


class UBIFSCompression(IntEnum):
    NONE = 0
    LZO = 1
    ZLIB = 2
    ZSTD = 3


class UBIFSHandler(StructHandler):
    NAME = "ubifs"

    _BIG_ENDIAN_MAGIC = 0x06_10_18_31
    # UBIFS_FORMAT_VERSION in the Linux kernel, fmt_version 4 images are
    # still common in the wild
    _UBIFS_FORMAT_VERSION = 5
    _UBIFS_SB_NODE_SIZE = 4096
    _UBIFS_MIN_JHEADS_CNT = 1

    # TODO: At the moment, we only match on the UBIFS superblock. Do we also want to account for
    # cases where the first node isn't a UBIFS superblock? Would such a layout actually be valid?
    # It might be valid to be flagged, but not necessarily to be extracted.
    #
    # Since we are running the handlers against every single match, regardless of whether a
    # previous chunk has already been established. That means that, for example, if we find a
    # superblock, and then many other kinds of nodes, it will take forever to run caculate_chunk()
    # against all the other nodes, and we waste loads of time and resources.

    # magic (4 bytes), 16 bytes, node type (1 byte, 0x06 is superblock),
    # group type (1 byte), 2 nulls.
    PATTERNS = [
        HexString("31 18 10 06 [16] 06 ( 00 | 01 | 02 ) 00 00"),  # LE
        HexString("06 10 18 31 [16] 06 ( 00 | 01 | 02 ) 00 00"),  # BE
    ]

    C_DEFINITIONS = r"""
        typedef struct ubifs_ch {
            uint32 magic;
            uint32 crc;
            uint64 sqnum;
            uint32 len;
            uint8 node_type;
            uint8 group_type;
            uint8 padding[2];
        } ubifs_ch_t;

        typedef struct ubifs_sb_node {
            ubifs_ch_t ch;
            uint8 padding[2];
            uint8 key_hash;
            uint8 key_fmt;
            uint32 flags;
            uint32 min_io_size;
            uint32 leb_size;
            uint32 leb_cnt;
            uint32 max_leb_cnt;
            uint64 max_bud_bytes;
            uint32 log_lebs;
            uint32 lpt_lebs;
            uint32 orph_lebs;
            uint32 jhead_cnt;
            uint32 fanout;
            uint32 lsave_cnt;
            uint32 fmt_version;
            uint16 default_compr;
            uint8 padding1[2];
            uint32 rp_uid;
            uint32 rp_gid;
            uint64 rp_size;
            uint32 time_gran;
            uint8 uuid[16];
            uint32 ro_compat_version;
            uint8 hmac[64];
            uint8 hmac_wkm[64];
            uint16 hash_algo;
            uint8 hash_mst[64];
            uint8 padding2[3774];
        } ubifs_sb_node_t;
    """
    HEADER_STRUCT = "ubifs_sb_node_t"

    EXTRACTOR = Command("ubireader_extract_files", "{inpath}", "-w", "-o", "{outdir}")

    DOC = HandlerDoc(
        name="UBIFS",
        description="UBIFS (Unsorted Block Image File System) is a flash file system designed for raw flash memory, providing wear leveling, error correction, and power failure resilience. It operates on top of UBI volumes, which manage flash blocks on raw NAND or NOR flash devices. The handler matches the UBIFS superblock node, validates its fields and covers the whole filesystem (LEB size times LEB count). Files are extracted with ubireader.",
        handler_type=HandlerType.FILESYSTEM,
        vendor=None,
        references=[
            Reference(
                title="UBIFS Documentation",
                url="https://www.kernel.org/doc/html/latest/filesystems/ubifs.html",
            ),
            Reference(
                title="UBIFS Wikipedia",
                url="https://en.wikipedia.org/wiki/UBIFS",
            ),
            Reference(
                title="UBIFS node definitions in the Linux kernel",
                url="https://github.com/torvalds/linux/blob/v6.12/fs/ubifs/ubifs-media.h",
            ),
        ],
        limitations=[
            "Only images starting with a superblock node are recognized.",
        ],
    )

    @classmethod
    def _validate_sb_node(cls, sb_header):
        if sb_header.ch.len != cls._UBIFS_SB_NODE_SIZE:
            raise InvalidInputFormat("Invalid UBIFS superblock node length.")

        try:
            UBIFSKeyHash(sb_header.key_hash)
            UBIFSKeyFormat(sb_header.key_fmt)
            UBIFSCompression(sb_header.default_compr)
        except ValueError as exc:
            raise InvalidInputFormat("Invalid UBIFS superblock field.") from exc

        if not 1 <= sb_header.fmt_version <= cls._UBIFS_FORMAT_VERSION:
            raise InvalidInputFormat("Unknown UBIFS format version.")

        if sb_header.jhead_cnt < cls._UBIFS_MIN_JHEADS_CNT:
            raise InvalidInputFormat("Invalid UBIFS journal head count.")

        min_io_size = sb_header.min_io_size
        if min_io_size == 0 or min_io_size & (min_io_size - 1):
            raise InvalidInputFormat("UBIFS min. I/O size is not a power of two.")

        if sb_header.leb_size == 0 or sb_header.leb_size % min_io_size:
            raise InvalidInputFormat("Invalid UBIFS LEB size.")

        if sb_header.leb_cnt == 0 or sb_header.leb_cnt > sb_header.max_leb_cnt:
            raise InvalidInputFormat("Invalid UBIFS LEB count.")

    def calculate_chunk(self, file: File, start_offset: int) -> ValidChunk | None:
        endian = get_endian(file, self._BIG_ENDIAN_MAGIC)
        sb_header = self.parse_header(file, endian)

        self._validate_sb_node(sb_header)

        # At the moment we are only matching on superblock nodes, so we can get the size of the
        # chunk from the LEB size * LEB count.
        ubifs_length = sb_header.leb_size * sb_header.leb_cnt

        return ValidChunk(
            start_offset=start_offset,
            end_offset=start_offset + ubifs_length,
        )


class UBIExtractor(Command):
    def extract(self, inpath: Path, outdir: Path):
        super().extract(inpath, outdir)
        # ubireader_extract_images creates a superfluous directory named
        # after the UBI file (inpath here), so we simply move the files up
        # and delete the remaining directory.
        superfluous_dir_path = outdir.joinpath(inpath.name)
        for file_path in superfluous_dir_path.iterdir():
            shutil.move(file_path.as_posix(), outdir.as_posix())
        shutil.rmtree(superfluous_dir_path.as_posix())


class UBIHandler(StructHandler):
    NAME = "ubi"

    _UBI_EC_HEADER_MAGIC = b"UBI#"
    _UBI_EC_MAGIC = 0x5542_4923  # "UBI#", big-endian
    _UBI_VERSION_1 = 1
    _UBI_EC_HEADER_SIZE = 64
    # The EC header CRC covers everything but the CRC field itself
    _UBI_EC_HEADER_CRC_OFFSET = 60

    # Number of EC header intervals used to guess the PEB size. Enough to
    # average out bad eraseblocks, small enough to stay within this image
    # when a blob contains multiple UBI images.
    _PEB_SIZE_GUESS_INTERVALS = 8

    # How many consecutive bad eraseblocks (PEBs without a valid EC header)
    # we skip over before deciding that the UBI image ended. The Linux UBI
    # bad eraseblock limit defaults to 20 (CONFIG_MTD_UBI_BEB_LIMIT).
    _MAX_CONSECUTIVE_BAD_PEBS = 20

    PATTERNS = [HexString("55 42 49 23 01  // UBI# and version 1")]

    # The UBI eraseblock counter (EC) header, stored big-endian at the
    # beginning of every PEB:
    # https://github.com/torvalds/linux/blob/v6.12/drivers/mtd/ubi/ubi-media.h
    C_DEFINITIONS = r"""
        typedef struct ubi_ec_header {
            uint32 magic;
            uint8 version;
            uint8 padding1[3];
            uint64 ec;
            uint32 vid_hdr_offset;
            uint32 data_offset;
            uint32 image_seq;
            uint8 padding2[32];
            uint32 hdr_crc;
        } ubi_ec_header_t;
    """
    HEADER_STRUCT = "ubi_ec_header_t"

    EXTRACTOR = UBIExtractor("ubireader_extract_images", "{inpath}", "-o", "{outdir}")

    DOC = HandlerDoc(
        name="UBI",
        description="UBI (Unsorted Block Image) is a volume management system for raw flash devices, providing wear leveling and bad block management. It operates as a layer between the MTD subsystem and higher-level filesystems like UBIFS. The handler walks the eraseblocks of the image, skipping bad eraseblocks with corrupted erase counter headers, and carves out the whole image. Volumes are extracted with ubireader into UBIFS images, which unblob then extracts recursively.",
        handler_type=HandlerType.FILESYSTEM,
        vendor=None,
        references=[
            Reference(
                title="UBI Documentation",
                url="https://www.kernel.org/doc/html/latest/driver-api/ubi.html",
            ),
            Reference(
                title="UBI Wikipedia",
                url="https://en.wikipedia.org/wiki/UBIFS#UBI",
            ),
            Reference(
                title="UBI header definitions in the Linux kernel",
                url="https://github.com/torvalds/linux/blob/v6.12/drivers/mtd/ubi/ubi-media.h",
            ),
        ],
        limitations=[
            "The PEB size is not stored in the image, it is guessed from the intervals between eraseblock headers.",
            "Eraseblocks are validated with their header CRC, images with systematically corrupted eraseblock headers are not recognized.",
        ],
    )

    def _parse_ec_header(self, file: File, offset: int):
        """Parse the EC header at offset, return None when it is not valid.

        A PEB with an invalid EC header (bad magic, version or CRC) is a bad
        eraseblock, this is how UBI itself detects them.
        """
        try:
            file.seek(offset)
        except SeekError:
            return None
        header_bytes = file.read(self._UBI_EC_HEADER_SIZE)
        if len(header_bytes) != self._UBI_EC_HEADER_SIZE:
            return None
        ec_header = self._struct_parser.parse(
            self.HEADER_STRUCT, header_bytes, Endian.BIG
        )
        if (
            ec_header.magic != self._UBI_EC_MAGIC
            or ec_header.version != self._UBI_VERSION_1
        ):
            return None
        crc = zlib.crc32(header_bytes[: self._UBI_EC_HEADER_CRC_OFFSET]) ^ 0xFFFFFFFF
        if crc != ec_header.hdr_crc:
            return None
        return ec_header

    def _find_image_ec_header_offsets(
        self, file: File, start_offset: int
    ) -> tuple[int, list[int]]:
        """Find the first few valid EC header offsets of this UBI image.

        Only EC headers of this image (same image sequence number) are
        considered, so that a blob containing multiple UBI images does not
        throw the PEB size guess off. Stops at the next UBI image, or when
        enough headers for the PEB size guess have been found.
        """
        candidate_offsets = list(iterate_patterns(file, self._UBI_EC_HEADER_MAGIC))
        image_seq = None
        image_offsets = []
        for offset in candidate_offsets:
            if offset < start_offset:
                continue
            ec_header = self._parse_ec_header(file, offset)
            if ec_header is None:
                continue
            if image_seq is None:
                image_seq = ec_header.image_seq
            elif ec_header.image_seq != image_seq:
                # Reached the next UBI image
                break
            image_offsets.append(offset)
            if len(image_offsets) > self._PEB_SIZE_GUESS_INTERVALS:
                break

        if image_seq is None:
            raise InvalidInputFormat("No valid UBI eraseblock headers found.")
        return image_seq, image_offsets

    def _guess_peb_size(self, file: File, start_offset: int) -> tuple[int, int]:
        """Guess the PEB size and the image sequence number of this UBI image.

        The PEB size is not stored anywhere in the image. Bad eraseblocks make
        the intervals between valid EC headers multiples of the PEB size, so
        we take the GCD of the first few intervals following start_offset.
        """
        image_seq, image_offsets = self._find_image_ec_header_offsets(
            file, start_offset
        )
        intervals = get_intervals(image_offsets)
        if not intervals:
            raise InvalidInputFormat(
                "UBI image has a single eraseblock, cannot guess the PEB size."
            )

        return math.gcd(*intervals), image_seq

    def _walk_ubi(self, file: File, peb_size: int, image_seq: int) -> int:
        """Walk PEBs from the current offset, skipping bad eraseblocks.

        PEBs without a valid EC header are bad eraseblocks: skip up to
        _MAX_CONSECUTIVE_BAD_PEBS of them in a row before deciding that the
        image ended. A valid EC header of another image (different image
        sequence number) also ends the walk. Returns the end offset of the
        last valid PEB, so that skipped bad eraseblocks stay inside the chunk
        and PEB numbering is preserved for the extractor.
        """
        consecutive_bad_pebs = 0
        end_offset = file.tell()
        first_peb = True
        while True:
            offset = file.tell()
            ec_header = self._parse_ec_header(file, offset)
            if ec_header is not None and ec_header.image_seq != image_seq:
                # This PEB belongs to another UBI image
                break
            try:
                file.seek(offset + peb_size)
            except SeekError:
                # The current PEB is truncated, the image ends before it
                break
            # The first PEB matched our pattern, so we know it belongs to the
            # image even when its header CRC does not check out.
            if first_peb or ec_header is not None:
                consecutive_bad_pebs = 0
                end_offset = offset + peb_size
            else:
                consecutive_bad_pebs += 1
                if consecutive_bad_pebs > self._MAX_CONSECUTIVE_BAD_PEBS:
                    break
            first_peb = False
        return end_offset

    def calculate_chunk(self, file: File, start_offset: int) -> ValidChunk | None:
        peb_size, image_seq = self._guess_peb_size(file, start_offset)

        logger.debug("Guessed UBI PEB size", size=peb_size, image_seq=image_seq)

        file.seek(start_offset)
        end_offset = self._walk_ubi(file, peb_size, image_seq)

        return ValidChunk(start_offset=start_offset, end_offset=end_offset)
