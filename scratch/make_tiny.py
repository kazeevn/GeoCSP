"""Tiny inits subset, to smoke-test the sampling path before committing to a long wait."""
import pickle

blob = pickle.load(open("/home/kna/DiffCSPNew/runs/bench/mp20/inits.pkl", "rb"))
pickle.dump(
    {
        "entries": blob["entries"],
        "trials": blob["trials"],
        "draws": [d for d in blob["draws"] if d["entry"] < 6],
    },
    open("scratch/inits_tiny.pkl", "wb"),
)
print("wrote scratch/inits_tiny.pkl")
