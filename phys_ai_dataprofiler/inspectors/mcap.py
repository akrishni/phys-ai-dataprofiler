from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse
from urllib.request import urlopen
from io import BytesIO

try:
    from mcap.reader import make_reader, FOOTER_SIZE, MAGIC_SIZE
    from mcap.stream_reader import StreamReader
    from mcap.records import Footer, Header, Channel as McapChannel, Schema
    MCAP_AVAILABLE = True
except ImportError:
    MCAP_AVAILABLE = False

MCAP_MAGIC = b"\x89MCAP0\r\n"
MAGIC_LEN = len(MCAP_MAGIC)


@dataclass(frozen=True)
class ChannelInfo:
    id: int
    topic: str
    message_encoding: str
    schema_id: int
    metadata: dict

@dataclass(frozen=True)
class SchemaInfo:
    id: int
    name: str
    encoding: str
    data: bytes

@dataclass(frozen=True)
class HeaderInfo:
    profile: str
    library: str

@dataclass(frozen=True)
class FooterInfo:
    summary_start: int
    summary_offset_start: int
    summary_crc: int

@dataclass(frozen=True)
class McapInspectionResult:
    url: str
    is_mcap: bool
    magic: bytes
    size_bytes: int | None
    path: Path | None = None
    error: str | None = None
    header: HeaderInfo | None = None
    footer: FooterInfo | None = None
    channels: list[ChannelInfo] | None = None
    schemas: list[SchemaInfo] | None = None
    topics: list[str] | None = None
    mcap_available: bool = MCAP_AVAILABLE


class McapFileInspector:
    """Inspect a file URL and verify it is a valid MCAP container."""

    def __init__(self, url: str) -> None:
        self.url = url

    def inspect(self) -> McapInspectionResult:
        try:
            path = self._resolve_local_path(self.url)
            if path is not None:
                return self._inspect_local(path)
            return self._inspect_remote(self.url)
        except OSError as exc:
            return McapInspectionResult(
                url=self.url,
                is_mcap=False,
                magic=b"",
                size_bytes=None,
                error=str(exc),
            )

    def _inspect_local(self, path: Path) -> McapInspectionResult:
        size = path.stat().st_size

        # Read magic bytes
        with path.open("rb") as f:
            magic = f.read(MAGIC_LEN)

        is_mcap = magic == MCAP_MAGIC

        # If mcap library is available and file is valid, parse with it
        if MCAP_AVAILABLE and is_mcap:
            try:
                with path.open("rb") as f:
                    mcap_data = self._parse_with_mcap(f)
                return McapInspectionResult(
                    url=self.url,
                    is_mcap=True,
                    magic=magic,
                    size_bytes=size,
                    path=path,
                    **mcap_data,
                )
            except Exception as e:
                return McapInspectionResult(
                    url=self.url,
                    is_mcap=True,
                    magic=magic,
                    size_bytes=size,
                    path=path,
                    error=f"Failed to parse with mcap library: {e}",
                )

        return McapInspectionResult(
            url=self.url,
            is_mcap=is_mcap,
            magic=magic,
            size_bytes=size,
            path=path,
        )

    def _inspect_remote(self, url: str) -> McapInspectionResult:
        # Fetch magic prefix (Range when the server supports it).
        req = __import__("urllib.request").Request(
            url,
            headers={"Range": f"bytes=0-{MAGIC_LEN - 1}"},
        )
        with urlopen(req) as resp:
            magic = resp.read(MAGIC_LEN)
            size = self._content_length(resp.headers)

        is_mcap = magic == MCAP_MAGIC

        # For remote files with mcap library, we need to download the entire file
        # (mcap library doesn't support streaming from HTTP range requests well)
        if MCAP_AVAILABLE and is_mcap and size:
            try:
                # Download entire file for parsing
                req = __import__("urllib.request").Request(url)
                with urlopen(req) as resp:
                    file_data = resp.read()

                mcap_data = self._parse_with_mcap(BytesIO(file_data))
                return McapInspectionResult(
                    url=url,
                    is_mcap=True,
                    magic=magic,
                    size_bytes=size,
                    **mcap_data,
                )
            except Exception as e:
                return McapInspectionResult(
                    url=url,
                    is_mcap=True,
                    magic=magic,
                    size_bytes=size,
                    error=f"Failed to parse with mcap library: {e}",
                )

        return McapInspectionResult(
            url=url,
            is_mcap=is_mcap,
            magic=magic,
            size_bytes=size,
        )

    @staticmethod
    def _resolve_local_path(url: str) -> Path | None:
        parsed = urlparse(url)
        if parsed.scheme in ("", "file"):
            # file:///tmp/foo.mcap or bare /tmp/foo.mcap
            raw = unquote(parsed.path) if parsed.scheme == "file" else url
            return Path(raw)
        return None

    @staticmethod
    def _parse_with_mcap(file_handle) -> dict:
        """Parse MCAP file using the mcap library to extract header, footer, and summary."""
        if not MCAP_AVAILABLE:
            return {"error": "mcap library not available"}

        reader = make_reader(file_handle)

        header_info = None
        footer_info = None
        channels = []
        schemas = []
        topics = set()

        # Get header information
        header = reader.get_header()
        if header:
            header_info = HeaderInfo(
                profile=header.profile,
                library=header.library,
            )

        # Read summary information
        summary = reader.get_summary()

        if summary:
            # Extract channels
            if summary.channels:
                for channel_id, channel in summary.channels.items():
                    channels.append(ChannelInfo(
                        id=channel.id,
                        topic=channel.topic,
                        message_encoding=channel.message_encoding,
                        schema_id=channel.schema_id,
                        metadata=channel.metadata,
                    ))
                    topics.add(channel.topic)

            # Extract schemas
            if summary.schemas:
                for schema_id, schema in summary.schemas.items():
                    schemas.append(SchemaInfo(
                        id=schema.id,
                        name=schema.name,
                        encoding=schema.encoding,
                        data=schema.data,
                    ))

        # Read the footer directly from the end of the file (Footer record +
        # closing magic) instead of streaming every record from the start.
        try:
            file_handle.seek(-(FOOTER_SIZE + MAGIC_SIZE), 2)
            stream_reader = StreamReader(file_handle, skip_magic=True)
            for record in stream_reader.records:
                if isinstance(record, Footer):
                    footer_info = FooterInfo(
                        summary_start=record.summary_start,
                        summary_offset_start=record.summary_offset_start,
                        summary_crc=record.summary_crc,
                    )
                    break
        except Exception:
            pass  # Footer not found or not seekable

        return {
            "header": header_info,
            "footer": footer_info,
            "channels": channels if channels else None,
            "schemas": schemas if schemas else None,
            "topics": sorted(list(topics)) if topics else None,
        }

    @staticmethod
    def _content_length(headers) -> int | None:
        # Prefer Content-Range total when using Range requests.
        content_range = headers.get("Content-Range")
        if content_range and "/" in content_range:
            total = content_range.rsplit("/", 1)[-1]
            if total.isdigit():
                return int(total)
        length = headers.get("Content-Length")
        return int(length) if length and length.isdigit() else None


if __name__ == "__main__":
    import sys

    result = McapFileInspector(sys.argv[1]).inspect()
    print(result)