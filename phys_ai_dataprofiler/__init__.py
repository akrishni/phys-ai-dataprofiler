"""phys_ai_dataprofiler: zero-copy profiling of physical-AI dataset files (MCAP, ROS bag, MATLAB, Parquet)."""

__version__ = "0.2.0"

from phys_ai_dataprofiler.inspectors.mcap import McapFileInspector, McapInspectionResult  # noqa: F401
from phys_ai_dataprofiler.inspectors.mat import MatFileInspector, MatInspectionResult  # noqa: F401
from phys_ai_dataprofiler.inspectors.rich import RichInspector, RichProfile  # noqa: F401
from phys_ai_dataprofiler.remote.range_file import HfRangeFile, LocalRangeFile, RangeFile  # noqa: F401
from phys_ai_dataprofiler.remote.inspector import inspect_mcap_remote, inspect_mat_remote, sniff_format  # noqa: F401
from phys_ai_dataprofiler.connectors.huggingface import HuggingFaceConnector  # noqa: F401
from phys_ai_dataprofiler.remote.parquet import inspect_parquet, ParquetInspectionResult  # noqa: F401
