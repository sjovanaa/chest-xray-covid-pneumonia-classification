"""
Poredjenje varijante sa isecanjem i bez isecanja.

Eksperiment proverava da li modeli donose odluku na osnovu plucnog tkiva ili
delimicno koriste obelezja izvan njega. Sa svakog snimka uklanja se pojas
odozgo (vrat i ramena), odozdo (trbuh) i uske trake sa strana, pa se isti
modeli obucavaju na tako pripremljenim snimcima.

Isecanje je identicno za sve tri klase i za sve podskupove, cime ostaje
jedina razlika izmedju dve varijante.

Skripta ne obucava modele - samo uparuje vec izracunate metrike i pravi
tabelu, grafikon i prikaz samog postupka isecanja.


"""
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from src.data import (_iseci, imena_klasa, indeksiraj_snimke, podeli_podatke,
                      ucitaj_konfiguraciju)
from src.paths import direktorijum_rezultata

plt.rcParams.update({"savefig.dpi": 300, "savefig.bbox": "tight", "font.size": 11,
                     "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})

NASTAVAK = " (isecen)"
METRIKE = ["Accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)",
           "Recall COVID19"]


def upari(tab):
    ceo = tab[~tab["Model"].str.endswith(NASTAVAK)].set_index("Model")
    isecen = tab[tab["Model"].str.endswith(NASTAVAK)].copy()
    isecen["osnovni"] = isecen["Model"].str.replace(NASTAVAK, "", regex=False)
    isecen = isecen.set_index("osnovni")

    redovi = []
    for ime in isecen.index:
        if ime not in ceo.index:
            continue
        red = {"Model": ime}
        for m in METRIKE:
            red[f"{m} - ceo"] = ceo.loc[ime, m]
            red[f"{m} - isecen"] = isecen.loc[ime, m]
            red[f"{m} - razlika"] = isecen.loc[ime, m] - ceo.loc[ime, m]
        redovi.append(red)
    return pd.DataFrame(redovi)


def grafikon(par, putanja):
    metrike = ["Accuracy", "F1 (macro)", "Recall COVID19"]
    modeli = par["Model"].tolist()
    x = np.arange(len(modeli))

    fig, ose = plt.subplots(1, len(metrike), figsize=(4.6 * len(metrike), 4.4),
                            sharey=True)
    for osa, m in zip(ose, metrike):
        osa.bar(x - 0.2, par[f"{m} - ceo"], 0.4, label="ceo snimak",
                color="#2980b9")
        osa.bar(x + 0.2, par[f"{m} - isecen"], 0.4, label="isecen snimak",
                color="#e67e22")
        osa.set_xticks(x, modeli, rotation=20, ha="right", fontsize=9)
        osa.set_title(m)
    ose[0].set_ylim(0.8, 1.0)
    ose[0].set_ylabel("Vrednost metrike")
    ose[0].legend(fontsize=9)

    fig.suptitle("Uticaj isecanja oblasti izvan grudnog kosa", fontsize=13)
    plt.tight_layout()
    plt.savefig(putanja)
    plt.close()


def prikaz_isecanja(cfg, putanja, po_klasi=1):
    df = indeksiraj_snimke()
    klase = imena_klasa(df)
    _, _, test = podeli_podatke(df, cfg["podaci"]["udeo_validacionog"], cfg["seed"])
    isecanje = dict(cfg["isecanje"], aktivno=True)

    primeri = []
    for k in klase:
        uzorak = test[test.klasa == k].sample(po_klasi, random_state=cfg["seed"])
        primeri += [(r.putanja, k) for _, r in uzorak.iterrows()]

    fig, ose = plt.subplots(2, len(primeri), figsize=(3.2 * len(primeri), 6.6))
    for j, (put, klasa) in enumerate(primeri):
        bajtovi = tf.io.read_file(put)
        original = tf.io.decode_image(bajtovi, channels=3, expand_animations=False)
        isecen = _iseci(original, isecanje)

        ose[0, j].imshow(original.numpy().astype("uint8"))
        ose[0, j].set_title(f"{klasa}\nceo snimak", fontsize=10)
        ose[1, j].imshow(isecen.numpy().astype("uint8"))
        ose[1, j].set_title("isecen snimak", fontsize=10)
        for osa in (ose[0, j], ose[1, j]):
            osa.axis("off")

    udeli = (f"gore {isecanje['gore']:.0%}, dole {isecanje['dole']:.0%}, "
             f"strane {isecanje['strane']:.0%}")
    fig.suptitle(f"Postupak isecanja ({udeli})", fontsize=13)
    plt.tight_layout()
    plt.savefig(putanja)
    plt.close()


def main():
    izlaz = direktorijum_rezultata()
    putanja_tabele = izlaz / "tabela_4_poredjenje_modela.csv"
    if not putanja_tabele.exists():
        print("Prvo pokreni scripts.evaluacija.")
        return

    tab = pd.read_csv(putanja_tabele)
    par = upari(tab)
    if par.empty:
        print("Nije pronadjen nijedan par (ceo, isecen). "
              "Pokreni trening sa zastavicom --isecanje.")
        return

    par.round(4).to_csv(izlaz / "tabela_8_isecanje.csv", index=False)
    grafikon(par, izlaz / "grafikon_isecanje.png")

    cfg = ucitaj_konfiguraciju()
    prikaz_isecanja(cfg, izlaz / "slika_isecanje.png")

    print("=" * 96)
    print("UTICAJ ISECANJA OBLASTI IZVAN GRUDNOG KOSA")
    print("=" * 96)
    prikaz = par[["Model"] + [f"{m} - {v}" for m in ["Accuracy", "F1 (macro)",
                                                     "Recall COVID19"]
                              for v in ["ceo", "isecen", "razlika"]]]
    print(prikaz.round(4).to_string(index=False))
    print("=" * 96)

    prosek = par[[f"{m} - razlika" for m in METRIKE]].mean()
    print("\nProsecna razlika (iseceno minus ceo), po svim modelima:")
    for m in METRIKE:
        print(f"  {m:<20} {prosek[f'{m} - razlika']:+.4f}")
    print("\nNegativna vrednost znaci da je varijanta sa isecanjem losija.")
    print(f"\nSacuvano u {izlaz}")


if __name__ == "__main__":
    main()
