#!/usr/bin/env python3
"""Create a sample MCAP file for testing."""

from mcap.writer import Writer
import io


def create_sample_mcap(output_path):
    """Create a sample MCAP file with channels and schemas."""

    with open(output_path, "wb") as f:
        writer = Writer(f)
        writer.start(profile="test", library="sample_creator")

        # Add a schema
        schema_id = writer.register_schema(
            name="example.Point",
            encoding="jsonschema",
            data=b'{"type": "object", "properties": {"x": {"type": "number"}, "y": {"type": "number"}}}',
        )

        # Add channels
        channel1_id = writer.register_channel(
            topic="/sensors/gps",
            message_encoding="json",
            schema_id=schema_id,
            metadata={"description": "GPS sensor data"},
        )

        channel2_id = writer.register_channel(
            topic="/sensors/imu",
            message_encoding="json",
            schema_id=schema_id,
            metadata={"description": "IMU sensor data"},
        )

        # Add some messages
        writer.add_message(
            channel_id=channel1_id,
            log_time=1000000000,
            data=b'{"x": 1.0, "y": 2.0}',
            publish_time=1000000000,
        )

        writer.add_message(
            channel_id=channel2_id,
            log_time=2000000000,
            data=b'{"x": 3.0, "y": 4.0}',
            publish_time=2000000000,
        )

        writer.finish()

    print(f"Created sample MCAP file: {output_path}")


if __name__ == "__main__":
    import sys
    output = sys.argv[1] if len(sys.argv) > 1 else "samples/sample.mcap"
    create_sample_mcap(output)
