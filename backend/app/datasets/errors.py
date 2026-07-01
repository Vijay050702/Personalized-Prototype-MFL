class DatasetError(Exception):
    pass


class DatasetNotFoundError(DatasetError):
    pass


class DatasetAlreadyExistsError(DatasetError):
    pass


class DatasetValidationError(DatasetError):
    pass


class DownloadError(DatasetError):
    pass


class ChecksumError(DownloadError):
    pass


class PartitionError(DatasetError):
    pass


class PreprocessingError(DatasetError):
    pass


class InvalidModalityError(DatasetError):
    pass


class CacheError(DatasetError):
    pass
