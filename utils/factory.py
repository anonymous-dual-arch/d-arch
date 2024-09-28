def get_model(model_name, args):
    name = model_name.lower()
    if name == "icarl":
        from models.icarl import iCaRL
        return iCaRL(args)
    elif name == "wa":
        from models.wa import WA
        return WA(args)
    elif name == "der":
        from models.der import DER
        return DER(args)
    elif name == "foster":
        from models.foster import FOSTER
        return FOSTER(args)
    elif name == "memo":   
        from models.memo import MEMO
        return MEMO(args)
    elif name == "icarl_t":
        from models.icarl_t import iCaRL
        return iCaRL(args)
    elif name == "wa_t":
        from models.wa_t import WA
        return WA(args)
    elif name == "foster_t":
        from models.foster_t import FOSTER
        return FOSTER(args)
    elif name == "der_t":
        from models.der_t import DER
        return DER(args)
    elif name == "memo_t":
        from models.memo_t import MEMO
        return MEMO(args)
    else:
        assert 0
