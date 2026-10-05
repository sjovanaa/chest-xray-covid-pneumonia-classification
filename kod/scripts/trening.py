"""
Obucavanje modela u dve faze.

  Faza 1 - osnova je zamrznuta, obucava se samo klasifikaciona glava
  Faza 2 - odmrzava se gornji deo osnove i fino podesava malom stopom ucenja

Model 'cnn_od_nule' nema pretrenirane tezine, pa se obucava u jednoj fazi
sa ukupnim brojem epoha obe faze.

"""
import argparse
import copy
import json
import os
import time
from datetime import datetime

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score

from src.data import pripremi_sve, tezine_klasa, ucitaj_konfiguraciju
from src.models import PUNA_IMENA, broj_parametara, napravi_model, odmrzni_osnovu
from src.paths import direktorijum_rezultata


class MakroF1(keras.callbacks.Callback):

    def __init__(self, validacioni):
        super().__init__()
        self.validacioni = validacioni
        self.y_stvarno = None      # racuna se jednom, ne u svakoj epohi

    def on_epoch_end(self, epoha, dnevnik=None):
        dnevnik = dnevnik if dnevnik is not None else {}
        if self.y_stvarno is None:
            self.y_stvarno = np.concatenate([o.numpy() for _, o in self.validacioni])
        y_pred = np.argmax(self.model.predict(self.validacioni, verbose=0), axis=1)
        dnevnik["val_makro_f1"] = f1_score(self.y_stvarno, y_pred, average="macro")
        print(f"   val_makro_f1: {dnevnik['val_makro_f1']:.4f}")


def _optimizator(stopa, t):
    clip = t.get("clipnorm", 0)
    dodatno = {"clipnorm": clip} if clip else {}
    if t.get("optimizator") == "adamw":
        return keras.optimizers.AdamW(
            stopa, weight_decay=t.get("opadanje_tezina", 1e-4), **dodatno)
    return keras.optimizers.Adam(stopa, **dodatno)


def _povratni_pozivi(validacioni, izlaz, strpljenje, faza):
    return [
        MakroF1(validacioni),
        keras.callbacks.EarlyStopping(
            monitor="val_makro_f1", mode="max", patience=strpljenje,
            restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_makro_f1", mode="max", factor=0.5,
            patience=max(2, strpljenje // 2), min_lr=1e-7, verbose=1),
        keras.callbacks.CSVLogger(izlaz / f"dnevnik_faza{faza}.csv"),
    ]


def obuci(ime, skupovi, tabele, klase, cfg, sufiks="", puno_ime=None):
    izlaz = direktorijum_rezultata() / (ime + sufiks)
    puno_ime = puno_ime or PUNA_IMENA[ime]
    izlaz.mkdir(parents=True, exist_ok=True)

    t = cfg["trening"]
    tezine = tezine_klasa(tabele["trening"], klase) if t["koristi_tezine_klasa"] else None

    print("\n" + "=" * 70)
    print(f"MODEL: {puno_ime}")
    print("=" * 70)

    model = napravi_model(ime, len(klase), cfg)
    param = broj_parametara(model)
    print(f"Parametara ukupno: {param['ukupno']:,} | obucivih: {param['obucivi']:,}")

    pocetak = time.time()
    istorija = {}

    if ime == "cnn_od_nule":
        epohe = t["epohe_glava"] + t["epohe_finog"]
        stopa = (t["lr_glava"] if "optimizator" in t
                 else t.get("lr_cnn", t["lr_glava"]))
        print(f"\nJednofazno obucavanje ({epohe} epoha, lr={stopa})")
    else:
        epohe = t["epohe_glava"]
        stopa = t["lr_glava"]
        print(f"\nFAZA 1 - zamrznuta osnova ({epohe} epoha, lr={stopa})")

    model.compile(optimizer=_optimizator(stopa, t),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    h1 = model.fit(skupovi["trening"], validation_data=skupovi["validacioni"],
                   epochs=epohe, class_weight=tezine,
                   callbacks=_povratni_pozivi(skupovi["validacioni"], izlaz,
                                              t["strpljenje"], 1),
                   verbose=1)
    istorija["faza1"] = {k: [float(v) for v in vr] for k, vr in h1.history.items()}
    najbolji_f1 = max(h1.history["val_makro_f1"])
    najbolje_tezine = model.get_weights()      # snimak najboljih tezina faze 1
    najbolja_faza = 1

    if ime != "cnn_od_nule":
        odmrznuto = odmrzni_osnovu(model, t.get("udeo_odmrznutih", 0.3))
        print(f"\nFAZA 2 - fino podesavanje: odmrznuto {odmrznuto} slojeva, "
              f"lr={t['lr_finog']}")

        model.compile(optimizer=_optimizator(t["lr_finog"], t),
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])

        h2 = model.fit(skupovi["trening"], validation_data=skupovi["validacioni"],
                       epochs=t["epohe_finog"], class_weight=tezine,
                       callbacks=_povratni_pozivi(skupovi["validacioni"], izlaz,
                                                  t["strpljenje"], 2),
                       verbose=1)
        istorija["faza2"] = {k: [float(v) for v in vr] for k, vr in h2.history.items()}

        f1_faza2 = max(h2.history["val_makro_f1"])
        if f1_faza2 > najbolji_f1:
            najbolji_f1, najbolja_faza = f1_faza2, 2
        else:
            print(f"\nFaza 2 ({f1_faza2:.4f}) nije nadmasila fazu 1 "
                  f"({najbolji_f1:.4f}); vracaju se tezine faze 1.")
            model.set_weights(najbolje_tezine)

    print(f"\nNajbolji val_makro_f1: {najbolji_f1:.4f} (faza {najbolja_faza})")
    trajanje = time.time() - pocetak

    model.save(izlaz / "model.keras")
    sazetak = {
        "model": ime + sufiks,
        "puno_ime": puno_ime,
        "najbolji_val_makro_f1": round(float(najbolji_f1), 4),
        "najbolja_faza": najbolja_faza,
        "parametri": param,
        "trajanje_sekundi": round(trajanje, 1),
        "trajanje_citljivo": f"{int(trajanje // 60)} min {int(trajanje % 60)} s",
        "vreme": datetime.now().isoformat(timespec="seconds"),
        "konfiguracija": cfg["trening"],
        "istorija": istorija,
    }
    with open(izlaz / "sazetak.json", "w", encoding="utf-8") as f:
        json.dump(sazetak, f, indent=2, ensure_ascii=False)

    print(f"\nZavrseno za {sazetak['trajanje_citljivo']}. Sacuvano u {izlaz}")
    return sazetak


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", help="ime jednog modela")
    p.add_argument("--svi", action="store_true", help="svi modeli iz config.yaml")
    p.add_argument("--brzo", action="store_true",
                   help="mali uzorak i 2 epohe - samo provera ispravnosti")
    p.add_argument("--isecanje", action="store_true",
                   help="obucava na isecenim snimcima (samo grudni kos)")
    p.add_argument("--optimizovan", action="store_true",
                   help="koristi hiperparametre pronadjene optimizacijom")
    args = p.parse_args()

    ogranici = 60 if args.brzo else None
    cfg_polazni = ucitaj_konfiguraciju()
    sufiks = "_opt" if args.optimizovan else ""
    if args.isecanje:
        cfg_polazni["isecanje"]["aktivno"] = True
        sufiks += "_isecen"
        print("Isecanje je ukljuceno:", cfg_polazni["isecanje"])
    skupovi, tabele, klase, cfg = pripremi_sve(cfg=cfg_polazni, ogranici=ogranici)

    if args.brzo:
        cfg["trening"]["epohe_glava"] = 2
        cfg["trening"]["epohe_finog"] = 1
        cfg["trening"]["strpljenje"] = 99

    print("GPU:", tf.config.list_physical_devices("GPU") or "nije dostupan (CPU)")
    print(f"Trening: {len(tabele['trening'])} | Validacioni: {len(tabele['validacioni'])}"
          f" | Test: {len(tabele['test'])}")

    modeli = cfg["modeli"] if args.svi else [args.model]
    if not modeli or modeli == [None]:
        p.error("navedi --model IME ili --svi")

    sazeci = []
    for ime in modeli:
        cfg_modela = copy.deepcopy(cfg)
        puno_ime = PUNA_IMENA[ime] + (" (isecen)" if args.isecanje else "")

        if args.optimizovan:
            putanja = direktorijum_rezultata() / f"{ime}_opt" / "sazetak.json"
            if not putanja.exists():
                print(f"Preskacem {ime}: nema optimizovane konfiguracije "
                      f"({putanja}).")
                continue
            sazetak_opt = json.loads(putanja.read_text(encoding="utf-8"))
            if not sazetak_opt.get("konfiguracija_treninga"):
                print(f"Preskacem {ime}: sazetak nema konfiguraciju treninga.")
                continue
            cfg_modela["trening"].update(sazetak_opt["konfiguracija_treninga"])
            puno_ime = (sazetak_opt.get("puno_ime", puno_ime)
                        + (" (isecen)" if args.isecanje else ""))

        sazeci.append(obuci(ime, skupovi, tabele, klase, cfg_modela,
                            sufiks, puno_ime))

    if not sazeci:
        print("Nijedan model nije obucen.")
        return

    print("\n" + "=" * 70)
    print("SVI MODELI ZAVRSENI")
    print("=" * 70)
    for s in sazeci:
        print(f"  {s['puno_ime']:<20} {s['trajanje_citljivo']}")


if __name__ == "__main__":
    main()
