"""Run with python3 test_scientific_audit.py; stdlib only, no UI imports."""
import ast
import math
from pathlib import Path

source = Path(__file__).with_name("sapi_weight_predictor.py").read_text()
data_names = {"ANIMAL_DATA", "ANIMAL_FORMULAS", "SLAUGHTER_DATA"}
function_names = {"hitung_berat_badan", "hitung_komponen_karkas", "calculate_error_range"}
nodes = []
for node in ast.parse(source).body:
    if isinstance(node, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id in data_names for t in node.targets
    ):
        nodes.append(node)
    elif isinstance(node, ast.FunctionDef) and node.name in function_names:
        nodes.append(node)
ns = {}
exec(compile(ast.Module(body=nodes, type_ignores=[]), "audit", "exec"), ns)


def rejects(function, *args):
    try:
        function(*args)
    except (ValueError, TypeError):
        return
    raise AssertionError(f"Invalid input accepted: {args}")


weight = ns["hitung_berat_badan"]
carcass = ns["hitung_komponen_karkas"]
bounds = ns["calculate_error_range"]
identity = ("Sapi", "Sapi Bali", "Jantan")
for invalid in (-1, 0, float("nan"), float("inf"), -float("inf"), True):
    rejects(weight, invalid, 150, *identity)
    rejects(weight, 170, invalid, *identity)
for invalid in (-1, float("nan"), float("inf"), True):
    rejects(carcass, invalid, *identity)
for invalid in (-1, 101, float("nan"), float("inf"), "bad"):
    rejects(bounds, 100, invalid)
assert all(math.isclose(actual, expected) for actual, expected in zip(bounds(100, 10), (90, 110)))
assert bounds(100, 0) == (100, 100)
assert carcass(0, *identity)["meat_weight"] == 0
for species, data in ns["ANIMAL_DATA"].items():
    for breed, profile in data["breeds"].items():
        ld = sum(profile["chest_range"].values()) / 2
        pb = sum(profile["length_range"].values()) / 2
        for sex in ("Jantan", "Betina"):
            value, _, _ = weight(ld, pb, species, breed, sex)
            assert math.isfinite(value) and value >= 0
            result = carcass(value, species, breed, sex)
            assert math.isclose(result["meat_weight"] + result["bone_and_fat_weight"], result["karkas_weight"])
            assert result["karkas_weight"] + sum(result["non_karkas_weights"].values()) <= value + 1e-9
# Reproduction, not validation: current NSA coefficient clips typical dimensions to zero.
assert ns["ANIMAL_FORMULAS"]["Domba"]["formulas"]["NSA Australia"]["calculation"](100, 100) == 0
print("PASS: invalid inputs rejected; arithmetic and carcass mass checks passed. Not biological validation.")
