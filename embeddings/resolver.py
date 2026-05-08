def get_encdec(encdec_type: str):
    if encdec_type == "BGE":
        from collective_encoder.nets.bge import BondGraphEncoderDecoder as EncDecModel
    else:
        raise ValueError("Unknown encoder type: " + encdec_type)
    return EncDecModel