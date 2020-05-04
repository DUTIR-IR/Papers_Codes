#    1. 放入字典调用

            def register(name=None, registry=None):
                def decorator(fn, registration_name=None):
                    module_name = registration_name or _default_name(fn)
                    if module_name in registry:
                        raise LookupError(f"module {module_name} already registered.")
                    registry[module_name] = fn
                    return fn
                return lambda fn: decorator(fn, name)
            
            registry = {}
            register = partial(register, registry=registry)
            @register('simple')
            class Encoder:
                // 内部实现
            
            
            from Encoders import registry as Encoder
            encoder_name = config["encoder"]
            encoder = Encoder[encoder_name](p1,p2,p3)
    
#    2. importlib
        encoder_name = config["encoder"]
        Encoder = importlib.import_module(encoder_name).component
        encoder = Encoder(p1,p2,p3)

