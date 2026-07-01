from math import ceil

from app.core.constants import DEFAULT_PAGE, DEFAULT_PAGE_SIZE


def paginate(
    total: int, page: int = DEFAULT_PAGE, page_size: int = DEFAULT_PAGE_SIZE
) -> dict:
    total_pages = ceil(total / page_size) if total > 0 else 1
    return {
        "page": page,
        "page_size": page_size,
        "total": total,
        "total_pages": total_pages,
        "has_next": page < total_pages,
        "has_previous": page > 1,
    }
