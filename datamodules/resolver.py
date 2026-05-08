from collective_encoder.datamodules.resolver import _REGISTRY, get_datamodule

_REGISTRY["SEQUENCE"] = ("datamodules.sequence", "SequenceDataModule")