from datetime import datetime


def format_datetime(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def iso_format(dt: datetime) -> str:
    return dt.isoformat()
