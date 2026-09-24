"""Run with python3 test_cattle_data.py; no Streamlit dependency required."""
import ast
from copy import deepcopy
from pathlib import Path

source = Path(__file__).with_name("sapi_weight_predictor.py").read_text()
tree = ast.parse(source)
names = {"ANIMAL_DATA", "ANIMAL_FORMULAS", "SLAUGHTER_DATA", "BREED_PRICE_FACTORS", "CATTLE_ESTIMATE_BASES"}
nodes = []
for node in tree.body:
    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in node.targets):
        nodes.append(node)
    elif isinstance(node, ast.For) and isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Attribute) and isinstance(node.iter.func.value, ast.Name) and node.iter.func.value.id == "CATTLE_ESTIMATE_BASES":
        nodes.append(node)
    elif isinstance(node, ast.FunctionDef) and node.name in {"hitung_berat_badan", "hitung_karkas"}:
        nodes.append(node)
namespace = {"deepcopy": deepcopy}
exec(compile(ast.Module(body=nodes, type_ignores=[]), "cattle_data", "exec"), namespace)
breeds = namespace["ANIMAL_DATA"]["Sapi"]["breeds"]
slaughter = namespace["SLAUGHTER_DATA"]["Sapi"]["breeds"]
assert len(breeds) == 40
assert breeds.keys() == slaughter.keys()
for name, base in namespace["CATTLE_ESTIMATE_BASES"].items():
    breed = breeds[name]
    assert "Estimasi belum tervalidasi" in breed["estimate_note"]
    assert base in slaughter[name]["reference"]
    assert breed["chest_range"] is not breeds[base]["chest_range"]
    assert slaughter[name]["karkas_percent"] is not slaughter[base]["karkas_percent"]
    assert namespace["BREED_PRICE_FACTORS"]["Sapi"][name] > 0
    for gender in ("Jantan", "Betina"):
        weight, _, _ = namespace["hitung_berat_badan"](
            sum(breed["chest_range"].values()) / 2,
            sum(breed["length_range"].values()) / 2,
            "Sapi", name, gender,
        )
        assert weight > 0, (name, gender)
        assert 0 < slaughter[name]["karkas_percent"][gender] < 100
print("PASS: 40 cattle entries; 32 labeled estimates; both genders calculate successfully")
