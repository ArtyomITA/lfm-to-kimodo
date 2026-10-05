# Encoder fantoccio: load_model di Kimodo esige di istanziare UN encoder; questo pesa zero.
# Viene sostituito subito dopo dallo stub LFM+mappa nel bridge.
class DummyEncoder:
    def __init__(self, **kwargs): pass
    def __call__(self, texts):
        raise RuntimeError("DummyEncoder: doveva essere sostituito dallo stub LFM")
    def to(self, d): return self
    def eval(self): return self
