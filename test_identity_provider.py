"""
Standalone-Test fuer identity_provider.py (Phase 1+2).
Braucht KEIN geladenes Model, KEIN echtes Sampling -- testet nur:
  - Modul-Import
  - Krea2Edit-Package-Discovery (_load_krea2edit_module)
  - Krea2EditProvider()-Instanziierung
  - get_default_provider()-Registry
  - max_refs-Deckelung (muss ValueError werfen, VOR jedem echten Forward-Call)
  - Fehlerpfade (leere refs, refs ohne 'latent')

Ausfuehren mit dem ComfyUI-eigenen Python (siehe run_test_identity_provider.bat).
"""
import os
import sys
import traceback

# ComfyUI-Root und RegioCraft-Ordner in den Pfad haengen, damit sowohl
# `import comfy...` (fuer den echten Krea2Edit-Import) als auch
# `import identity_provider` direkt funktionieren, unabhaengig vom
# ComfyUI-eigenen Custom-Node-Loader.
COMFYUI_ROOT = r"F:\ComfyUI\ComfyUI"
REGIOCRAFT_DIR = os.path.join(COMFYUI_ROOT, "custom_nodes", "RegioCraft")
sys.path.insert(0, COMFYUI_ROOT)
sys.path.insert(0, REGIOCRAFT_DIR)

results = []


def check(name, fn):
    try:
        fn()
        results.append((name, True, ""))
        print(f"[PASS] {name}")
    except Exception as e:
        results.append((name, False, f"{type(e).__name__}: {e}"))
        print(f"[FAIL] {name} -> {type(e).__name__}: {e}")
        traceback.print_exc()


# ---------------------------------------------------------------------------
def test_import():
    import identity_provider as ip
    assert hasattr(ip, "IdentityProvider")
    assert hasattr(ip, "Krea2EditProvider")
    assert hasattr(ip, "get_default_provider")


def test_krea2edit_discovery():
    import identity_provider as ip
    mod = ip._load_krea2edit_module()
    assert mod is not None, "comfyui-krea2edit module not found -- check folder name/path"
    assert hasattr(mod, "krea2_edit_forward"), "module found but krea2_edit_forward missing"


def test_provider_instantiation():
    import identity_provider as ip
    provider = ip.Krea2EditProvider()
    assert provider.max_refs == 2
    assert callable(provider.forward)


def test_registry():
    import identity_provider as ip
    provider = ip.get_default_provider("krea2edit")
    assert provider is not None, "registry returned None -- Krea2EditProvider() failed silently"
    assert isinstance(provider, ip.Krea2EditProvider)

    unknown = ip.get_default_provider("does_not_exist")
    assert unknown is None, "unknown provider name should return None, not raise"


def test_cap_enforcement():
    import identity_provider as ip
    provider = ip.Krea2EditProvider()
    fake_refs = [{"latent": {"samples": None}} for _ in range(3)]  # 3 > max_refs=2
    try:
        provider.forward(None, None, None, None, transformer_options={}, refs=fake_refs)
        raise AssertionError("expected ValueError for >max_refs, none was raised")
    except ValueError as e:
        assert "at most" in str(e)


def test_empty_refs_raises():
    import identity_provider as ip
    provider = ip.Krea2EditProvider()
    try:
        provider.forward(None, None, None, None, transformer_options={}, refs=[])
        raise AssertionError("expected ValueError for empty refs, none was raised")
    except ValueError:
        pass


def test_neither_image_nor_latent_raises():
    """2026-08-02 update: image-only refs are now IMPLEMENTED (pixel/fit path,
    blur fix) as long as a vae is also passed to forward(). This test now checks
    the remaining hard error case: a ref with NEITHER 'image' nor 'latent' --
    that must still raise a clear ValueError, not silently do nothing."""
    import identity_provider as ip
    provider = ip.Krea2EditProvider()
    try:
        provider.forward(None, None, None, None, transformer_options={},
                         refs=[{"boost": 1.0}])  # neither key present
        raise AssertionError("expected ValueError for a ref with no image/latent, none was raised")
    except ValueError:
        pass


# ---------------------------------------------------------------------------
print("=" * 70)
print("identity_provider.py -- Phase 1+2 standalone test")
print("=" * 70)

check("1) Modul importierbar", test_import)
check("2) comfyui-krea2edit gefunden + krea2_edit_forward vorhanden", test_krea2edit_discovery)
check("3) Krea2EditProvider() instanziiert", test_provider_instantiation)
check("4) get_default_provider() Registry funktioniert", test_registry)
check("5) max_refs=2 Deckelung wirft ValueError bei 3 Refs", test_cap_enforcement)
check("6) leere refs-Liste wirft ValueError", test_empty_refs_raises)
check("7) ref ohne image UND ohne latent wirft ValueError", test_neither_image_nor_latent_raises)

print("=" * 70)
n_pass = sum(1 for _, ok, _ in results if ok)
n_total = len(results)
print(f"ERGEBNIS: {n_pass}/{n_total} Tests bestanden")
if n_pass < n_total:
    print("Fehlgeschlagene Tests:")
    for name, ok, err in results:
        if not ok:
            print(f"  - {name}: {err}")
print("=" * 70)

sys.exit(0 if n_pass == n_total else 1)
