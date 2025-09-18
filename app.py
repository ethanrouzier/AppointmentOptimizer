#!/usr/bin/env python3
import os
import re
from datetime import datetime
from itertools import combinations
from datetime import date

import numpy as np
import pandas as pd
from flask import Flask, render_template, request
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import pgeocode

# ========== Config ==========
EXCEL_PATH = "data/rdv.csv"
DATE_COL = "DATE_RDV"
MOTIF_COL = "RDV"
NIP_COL = "NIP"
POSTAL_COL = "CODEPOSTALCODECOMMUNE"  # tel que fourni
CITY_COL = "VILLE"

MAX_PAIR_TIME_DIFF_MIN = 60
MAX_PAIR_DIST_KM = 10.0
DEFAULT_SITE_ONEWAY_KM = 10.0  # distance aller typique jusqu'au site
EMISSION_KG_PER_KM = 0.2       # ~0.2 kgCO2/km

# ========== App ==========
app = Flask(__name__, template_folder="templates", static_folder="static")
os.makedirs(os.path.join(app.static_folder, "plots"), exist_ok=True)

nomi = pgeocode.Nominatim("fr")

# --- Référence site : Caen (14000) ---
CAEN_POSTAL = "14000"
try:
    _caen = nomi.query_postal_code(CAEN_POSTAL)
    CAEN_LAT = float(_caen["latitude"])
    CAEN_LON = float(_caen["longitude"])
except Exception:
    # fallback approximatif (centre de Caen) si jamais pgeocode renvoie NaN
    CAEN_LAT = 49.1829
    CAEN_LON = -0.3707

ALLOWED_DEPTS = {"50", "61"}

def get_dept_from_cp(value):
    """Retourne le code département à partir du code postal (gère 97x/98x aussi)."""
    cp = extract_postal_code(value)
    if pd.isna(cp):
        return np.nan
    s = str(cp)
    if (s.startswith("97") or s.startswith("98")) and len(s) >= 3:
        return s[:3]   # ex: 971, 972...
    return s[:2]       # ex: 50, 61, 14...

def dist_to_caen_km(lat, lon):
    return haversine_km(lat, lon, CAEN_LAT, CAEN_LON)

def select_first_last(df):
    """
    Garde au plus 2 lignes par NIP pour la journée :
    - ALLER = premier RDV (heure min)
    - RETOUR = dernier RDV (heure max)
    Si un patient n’a qu’un seul RDV, il apparaît dans les deux trajets avec la même heure.
    """
    if df.empty:
        return df.copy()

    base = df.copy()
    # indices des premières/dernières occurrences par NIP
    idx_first = base.groupby(NIP_COL)[DATE_COL].idxmin()
    idx_last  = base.groupby(NIP_COL)[DATE_COL].idxmax()

    first_df = base.loc[idx_first].copy()
    first_df["trajet"] = "ALLER"

    last_df = base.loc[idx_last].copy()
    last_df["trajet"] = "RETOUR"

    # on autorise le cas "même index" (1 seul RDV) -> deux lignes ALLER/RETOUR identiques
    out = pd.concat([first_df, last_df], ignore_index=True)
    return out

def extract_postal_code(value):
    """Récupère un code postal FR à partir de valeurs type '50250,273' ou float."""
    if pd.isna(value):
        return np.nan
    s = str(value).strip()

    # cas float avec virgule décimale française
    if "," in s:
        s = s.split(",")[0]
    elif "." in s:  # au cas où ce soit un float avec un point
        s = s.split(".")[0]

    # garder que des chiffres
    digits = "".join([c for c in s if c.isdigit()])

    if len(digits) >= 5:
        return digits[:5]  # les 5 premiers chiffres suffisent
    return np.nan



def add_latlon(df):
    """Ajoute latitude/longitude depuis le code postal en mappant les uniques."""
    df = df.copy()
    df["__cp__"] = df[POSTAL_COL].apply(extract_postal_code)

    uniq = sorted([cp for cp in df["__cp__"].dropna().unique() if isinstance(cp, str) and len(cp) == 5])
    if len(uniq) == 0:
        raise ValueError("Aucun code postal valide (5 chiffres) n'a été détecté.")

    geo = nomi.query_postal_code(uniq)
    # garde seulement les lignes géocodées valides
    geo = geo[["postal_code", "latitude", "longitude"]].dropna(subset=["latitude", "longitude"])
    geo = geo.set_index("postal_code")

    df["lat"] = df["__cp__"].map(geo["latitude"])
    df["lon"] = df["__cp__"].map(geo["longitude"])
    df = df.drop(columns="__cp__", errors="ignore")

    # on supprime les lignes non géocodées pour éviter les surprises ensuite
    df = df.dropna(subset=["lat", "lon"])
    if df.empty:
        raise ValueError("Aucun point géocodé. Vérifie la colonne CODEPOSTALCODECOMMUNE (codes à 5 chiffres).")
    return df



def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2 +
         np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) *
         np.sin(dlon / 2) ** 2)
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def minutes_since_midnight(ts: pd.Timestamp) -> int:
    return int(ts.hour * 60 + ts.minute)


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))
        self.r = [0] * n

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.r[ra] < self.r[rb]:
            self.p[ra] = rb
        elif self.r[ra] > self.r[rb]:
            self.p[rb] = ra
        else:
            self.p[rb] = ra
            self.r[ra] += 1


def cluster_by_distance(df_motif):
    """Regroupe par proximité (≤10 km) via union-find, renvoie un vecteur de labels."""
    n = len(df_motif)
    uf = UnionFind(n)
    lats = df_motif["lat"].values
    lons = df_motif["lon"].values
    for i in range(n):
        for j in range(i + 1, n):
            if haversine_km(lats[i], lons[i], lats[j], lons[j]) <= MAX_PAIR_DIST_KM:
                uf.union(i, j)
    labels = [uf.find(i) for i in range(n)]
    # compact labels
    map_id = {old: k for k, old in enumerate(sorted(set(labels)))}
    return np.array([map_id[x] for x in labels], dtype=int)

def compute_affected_nips(pairs_before: pd.DataFrame, pairs_after: pd.DataFrame):
    """
    Renvoie l'ensemble des NIP (en str) impliqués dans des paires :
    - apparues (présentes après, absentes avant)
    - modifiées (partenaire différent pour le même trajet)
    """
    def to_pairs(df):
        if df is None or df.empty:
            return set()
        out = set()
        for _, r in df.iterrows():
            a, b = str(r["NIP_1"]), str(r["NIP_2"])
            traj = r.get("Trajet", "")
            out.add((traj, tuple(sorted([a, b]))))
        return out

    def partners_map(df):
        mp = {}
        if df is None or df.empty:
            return mp
        for _, r in df.iterrows():
            a, b = str(r["NIP_1"]), str(r["NIP_2"])
            traj = r.get("Trajet", "")
            mp.setdefault((traj, a), set()).add(b)
            mp.setdefault((traj, b), set()).add(a)
        return mp

    before_pairs = to_pairs(pairs_before)
    after_pairs  = to_pairs(pairs_after)

    # paires nouvelles
    new_pairs = after_pairs - before_pairs
    affected = set()
    for traj, (a, b) in new_pairs:
        affected.add(a); affected.add(b)

    # partenaires changés (même trajet)
    pb = partners_map(pairs_before)
    pa = partners_map(pairs_after)
    for (traj, a), partners_after in pa.items():
        partners_before = pb.get((traj, a), set())
        if partners_before and partners_after != partners_before:
            affected.add(a)
            affected.update(partners_after)

    return affected

def optimize_day(df_day):
    """Optimise les horaires dans chaque motif pour maximiser le Δ kgCO₂
       (via la somme des Saving_km après optimisation), sans modifier
       les patient·e·s ayant plusieurs RDV dans la journée (verrouillés)."""
    df_opt = df_day.copy().reset_index(drop=True)
    df_opt["time"] = df_opt[DATE_COL].dt.floor("min")  # minutes entières

    # NIP verrouillés = >1 RDV sur la journée
    counts = df_opt.groupby(NIP_COL).size()
    locked_nips = set(counts[counts > 1].index)

    # petit helper pour scorer l'état courant (somme des Saving_km)
    def current_saving_km():
        pairs = enumerate_pairs(df_opt)
        if pairs.empty or "Saving_km" not in pairs.columns:
            return 0.0
        return float(np.clip(pairs["Saving_km"].values, a_min=0, a_max=None).sum())

    # score de base
    best_global = current_saving_km()

    for motif, block in df_opt.groupby(MOTIF_COL, group_keys=False):
        idx = block.index.to_list()
        sub = df_opt.loc[idx].reset_index(drop=False)  # garde l'index global
        if len(sub) <= 1:
            continue

        # marquer verrouillés / flexibles
        sub["locked"] = sub[NIP_COL].isin(locked_nips)

        # clusters (pour un ordre stable et géo-logique)
        labels = cluster_by_distance(sub)
        sub["cluster"] = labels

        # tous les créneaux de ce motif
        slots = sorted(sub["time"].tolist())

        # séparer
        locked_rows = sub[sub["locked"]].copy()
        flex_rows   = sub[~sub["locked"]].copy()

        # retirer les créneaux déjà pris par les verrouillés
        remaining_slots = slots.copy()
        for t in locked_rows["time"]:
            try:
                remaining_slots.remove(t)
            except ValueError:
                pass

        # ordre d'essai pour les flex : par cluster, puis heure
        flex_sorted = flex_rows.sort_values(["cluster", "time", "lat", "lon"]).reset_index(drop=True)

        # pour chaque flexible, choisir le créneau qui maximise le score global
        for _, row in flex_sorted.iterrows():
            gidx = int(row["index"])                      # index global dans df_opt
            original_time = df_opt.at[gidx, DATE_COL]     # Timestamp
            best_time = original_time
            best_score = best_global

            # on teste tous les créneaux restants
            for t in list(remaining_slots):
                t = pd.Timestamp(t)
                # assignation temporaire
                df_opt.at[gidx, DATE_COL] = t
                df_opt.at[gidx, "time"]   = t

                new_score = current_saving_km()
                if new_score > best_score + 1e-9:
                    best_score = new_score
                    best_time = t

                # revert pour le prochain essai
                df_opt.at[gidx, DATE_COL] = original_time
                df_opt.at[gidx, "time"]   = original_time

            # commit du meilleur créneau trouvé
            df_opt.at[gidx, DATE_COL] = best_time
            df_opt.at[gidx, "time"]   = best_time
            best_global = best_score

            # garantir l'unicité des créneaux : on retire celui qu'on vient d'utiliser s'il est dispo
            try:
                remaining_slots.remove(best_time)
            except ValueError:
                pass  # il avait peut-être déjà été retiré (doublons de timestamps, etc.)

    return df_opt


def enumerate_pairs(df):
    """
    Paires admissibles (≤10 km & ≤60 min) en ne regardant que :
      - l'ALLER (premier RDV du NIP)
      - le RETOUR (dernier RDV du NIP)
    Matching glouton par trajet qui MAXIMISE la somme des km économisés selon :
      saving_km = max(0, 2 * min(dist(NIP_i, Caen), dist(NIP_j, Caen)) - dist(domiciles))
    """
    rows = []
    sel = select_first_last(df)

    for trajet in ["ALLER", "RETOUR"]:
        sub = sel[sel["trajet"] == trajet].reset_index(drop=True)
        if len(sub) < 2:
            continue

        candidates = []
        for i in range(len(sub)):
            for j in range(i + 1, len(sub)):
                nip_i = sub.loc[i, NIP_COL]
                nip_j = sub.loc[j, NIP_COL]
                if nip_i == nip_j:
                    continue  # jamais soi-même

                # distance entre domiciles
                dist_homes = haversine_km(sub.loc[i, "lat"], sub.loc[i, "lon"],
                                          sub.loc[j, "lat"], sub.loc[j, "lon"])
                if dist_homes > MAX_PAIR_DIST_KM:
                    continue

                # contrainte horaire
                t1 = sub.loc[i, DATE_COL]
                t2 = sub.loc[j, DATE_COL]
                dt_min = abs(int((t2 - t1).total_seconds() // 60))
                if dt_min > MAX_PAIR_TIME_DIFF_MIN:
                    continue

                # distance à Caen du plus proche du couple
                d_caen_i = dist_to_caen_km(sub.loc[i, "lat"], sub.loc[i, "lon"])
                d_caen_j = dist_to_caen_km(sub.loc[j, "lat"], sub.loc[j, "lon"])
                d_near = min(d_caen_i, d_caen_j)

                # km économisés (aller simple x2 pour la personne la plus proche, moins détour domicile-domicile)
                saving_km = max(0.0, 2.0 * d_near - dist_homes)

                # garder même si gain=0 (au cas où tu veux les voir), mais on optimisera par gain décroissant
                candidates.append((saving_km, dt_min, dist_homes, i, j))

        # tri : on maximise saving_km (desc), puis on préfère petit Δt, puis petite distance entre domiciles
        candidates.sort(key=lambda x: (-x[0], x[1], x[2]))

        used = set()
        for saving_km, dt_min, dist_homes, i, j in candidates:
            nip_i = sub.loc[i, NIP_COL]
            nip_j = sub.loc[j, NIP_COL]
            if nip_i in used or nip_j in used:
                continue

            rows.append({
                "Trajet": trajet,
                "NIP_1": nip_i,
                "NIP_2": nip_j,
                "Motif_1": sub.loc[i, MOTIF_COL],
                "Motif_2": sub.loc[j, MOTIF_COL],
                "Heure_1": sub.loc[i, DATE_COL].strftime("%H:%M"),
                "Heure_2": sub.loc[j, DATE_COL].strftime("%H:%M"),
                "Delta_min": int(dt_min),
                "Distance_km": round(float(dist_homes), 2),
                "Saving_km": round(float(saving_km), 2),
            })
            used.add(nip_i)
            used.add(nip_j)

    cols = ["Trajet","NIP_1","NIP_2","Motif_1","Motif_2",
            "Heure_1","Heure_2","Delta_min","Distance_km","Saving_km"]
    return pd.DataFrame(rows, columns=cols)




def estimate_co2_saving(pairs_df, _site_oneway_km_ignored=0.0):
    """
    Somme des 'Saving_km' si présent (calcul Caen), sinon retombe sur l'ancien calcul.
    """
    if pairs_df is None or pairs_df.empty:
        return 0.0, 0.0

    if "Saving_km" in pairs_df.columns:
        saved_km = float(np.clip(pairs_df["Saving_km"].values, a_min=0, a_max=None).sum())
    else:
        # fallback ancien comportement (au cas où)
        saved_km = float(np.clip(2 * DEFAULT_SITE_ONEWAY_KM - pairs_df["Distance_km"].values, a_min=0, a_max=None).sum())

    saved_kg = saved_km * EMISSION_KG_PER_KM
    return saved_km, float(saved_kg)



def make_3d_plot(df, title, out_path, allowed_nips=None):
    """
    Nuage 3D (lon, lat, heure). Couleur = motif.
    Si allowed_nips est fourni, on ne garde que ces NIP (comparés en str).
    """
    if allowed_nips is not None:
        keep = df[NIP_COL].astype(str).isin(set(allowed_nips))
        df = df.loc[keep].copy()

    if df.empty:
        plt.figure(figsize=(6, 4), dpi=140)
        plt.text(0.5, 0.5, "Aucune donnée (paires nouvelles/modifiées)", ha="center", va="center")
        plt.axis("off")
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        return

    # Axes
    fig = plt.figure(figsize=(7.5, 5.5), dpi=140)
    ax = fig.add_subplot(111, projection="3d")
    # Encodage temps en minutes
    tmin = df[DATE_COL].dt.hour * 60 + df[DATE_COL].dt.minute
    # Motifs -> ints
    motifs = df[MOTIF_COL].astype(str)
    uniq = {m: i for i, m in enumerate(sorted(motifs.unique()))}
    colors = [uniq[m] for m in motifs]

    p = ax.scatter(df["lon"], df["lat"], tmin, c=colors, s=38, depthshade=True)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_zlabel("Heure (min)")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    results = None
    selected_date = None
    site_km = DEFAULT_SITE_ONEWAY_KM

    if request.method == "POST":
        try:
            selected_date = request.form.get("date")
            site_km = float(request.form.get("site_km", DEFAULT_SITE_ONEWAY_KM))
            if not selected_date:
                raise ValueError("Choisis une date.")

            if not os.path.exists(EXCEL_PATH):
                raise FileNotFoundError(f"Excel introuvable : {EXCEL_PATH}")

            df = pd.read_csv(EXCEL_PATH, sep=";", dtype=str)
            # Filtre départements 50 & 61
            df["DEPT"] = df[POSTAL_COL].apply(get_dept_from_cp)
            df = df[df["DEPT"].isin(ALLOWED_DEPTS)].copy()
            if df.empty:
                raise ValueError("Aucun RDV dans les départements 50/61 (selon la colonne CODEPOSTALCODECOMMUNE).")

            if DATE_COL not in df.columns:
                raise KeyError(f"Colonne manquante : {DATE_COL}")
            if MOTIF_COL not in df.columns:
                raise KeyError(f"Colonne manquante : {MOTIF_COL}")
            if NIP_COL not in df.columns:
                raise KeyError(f"Colonne manquante : {NIP_COL}")
            if POSTAL_COL not in df.columns:
                raise KeyError(f"Colonne manquante : {POSTAL_COL}")

            # Parse dates
            df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce", dayfirst=True)
            df = df.dropna(subset=[DATE_COL])

            # Filtre sur le jour sélectionné
            day = pd.to_datetime(selected_date).date()
            day = pd.to_datetime(selected_date).date()
            if not (date(2022, 1, 1) <= day <= date(2022, 12, 31)):
                raise ValueError("Sélectionne une date entre le 01/01/2022 et le 31/12/2022.")
            df_day = df[df[DATE_COL].dt.date == day].copy()
            if df_day.empty:
                raise ValueError("Aucun RDV ce jour-là dans l'Excel.")

            # Géocodage CP -> lat/lon
            df_day = add_latlon(df_day)
            if df_day.empty:
                raise ValueError("Impossible d'obtenir des coordonnées (codes postaux manquants/invalides).")

            # Sauvegarde avant optim
            df_before = df_day.copy()

            # Optimisation (permutations au sein d'un même motif)
            df_after = optimize_day(df_day)

            # Paires avant/après
            pairs_before = enumerate_pairs(df_before)
            pairs_after = enumerate_pairs(df_after)
            affected_nips = compute_affected_nips(pairs_before, pairs_after)
            # Gains CO2
            # Gains CO2 — bénéfice dû aux échanges (Δ = après − avant)
            saved_km_before, saved_kg_before = estimate_co2_saving(pairs_before, site_km)
            saved_km_after,  saved_kg_after  = estimate_co2_saving(pairs_after, site_km)

            delta_km = saved_km_after - saved_km_before
            delta_kg = saved_kg_after - saved_kg_before

            # Graphiques 3D
            stamp = datetime.now().strftime("%Y%m%d%H%M%S")
            before_name = f"plots/before_{stamp}.png"
            after_name  = f"plots/after_{stamp}.png"
            make_3d_plot(df_before, f"Avant optimisation — {day.isoformat()} (paires modifiées/nouvelles)",
                        os.path.join(app.static_folder, before_name), allowed_nips=affected_nips)
            make_3d_plot(df_after,  f"Après optimisation — {day.isoformat()} (paires modifiées/nouvelles)",
                        os.path.join(app.static_folder, after_name),  allowed_nips=affected_nips)


            # Prépare résultats pour le template
            def df_to_records(d):
                return [] if d is None or d.empty else d.to_dict(orient="records")
            pairs_after_sorted = pairs_after.sort_values(
                ["Trajet", "Delta_min", "Distance_km"]
            ) if not pairs_after.empty else pairs_after
            results = {
                "count_rdv": int(len(df_day)),
                "count_pairs_before": int(len(pairs_before)),
                "count_pairs_after": int(len(pairs_after)),
                "delta_pairs": int(len(pairs_after) - len(pairs_before)),
                "saved_km": round(delta_km, 1),
                "saved_kg": round(delta_kg, 2),
                    "pairs_after": df_to_records(pairs_after_sorted),
                "before_plot": before_name,
                "after_plot": after_name,
            }

        except Exception as e:
            error = str(e)

    return render_template(
        "index.html",
        error=error,
        results=results,
        default_date="2022-01-19",
        selected_date=selected_date,
        site_km=site_km,
        defaults={"site_km": DEFAULT_SITE_ONEWAY_KM,
                  "emission": EMISSION_KG_PER_KM,
                  "max_dist": MAX_PAIR_DIST_KM,
                  "max_dt": MAX_PAIR_TIME_DIFF_MIN}
    )


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=4000)
