#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integra o dataset de microbiologia do solo (Schnecker et al. 2023, Geoderma 440, 116693)
com o clima histórico da Open-Meteo (reanálise ERA5).

O QUE O SCRIPT FAZ
  1. Lê dados_solo_austria_medias.csv (uma linha por local, tratamento e data).
  2. Baixa, para cada local, o clima horário de (primeira coleta - 90 dias) até a última coleta
     e guarda o resultado em cache (não repete a chamada se rodar de novo).
  3. Resume em valores diários e calcula janelas de 7, 14, 30, 60 e 90 dias ANTES de cada coleta
     (a janela termina no dia anterior à coleta).
  4. VALIDAÇÃO A: compara temperatura e umidade do solo da Open-Meteo com as medidas do artigo.
  5. VALIDAÇÃO B: refaz a regressão respiração x temperatura (Q10) e compara com a Figura 4 do artigo.
  6. Resume a precipitação de junho a outubro de 2018 e 2019 (o artigo cita 2018 como seco).

COMO RODAR
  pip install pandas numpy requests
  python clima_open_meteo.py
  (opções: --csv caminho.csv  --out pasta_de_saida  --refresh para ignorar o cache)

OBSERVAÇÕES
  - Variáveis horárias pedidas: temperature_2m, precipitation, et0_fao_evapotranspiration,
    soil_temperature_0_to_7cm, soil_moisture_0_to_7cm. Se a API recusar algum nome, confira a
    lista em https://open-meteo.com/en/docs/historical-weather-api e ajuste a lista HOURLY abaixo.
  - O uso gratuito da Open-Meteo é para fins não comerciais. Cite a fonte no TCC.
  - As coordenadas dos locais têm precisão de ~1 minuto de arco (vieram do corpo do artigo).
  - Os dados do artigo são MÉDIAS por data e tratamento (n = 16 datas por local e tratamento).
    O Q10 aqui é, por isso, só uma conferência aproximada do publicado (que usou amostras individuais).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

URL = "https://archive-api.open-meteo.com/v1/archive"
HOURLY = [
    "temperature_2m",
    "precipitation",
    "et0_fao_evapotranspiration",
    "soil_temperature_0_to_7cm",
    "soil_moisture_0_to_7cm",
]
WINDOWS = [7, 14, 30, 60, 90]
TIMEZONE = "Europe/Vienna"   # os dois locais ficam na Áustria
DRY_DAY_MM = 1.0             # dia "sem chuva" = precipitação < 1 mm

# Valores publicados na Figura 4 do artigo (respiração x temperatura do solo)
PUBLISHED_Q10 = {
    ("agricultural", "control"): {"r2": 0.73, "q10": 2.5},
    ("agricultural", "residue_removal"): {"r2": 0.79, "q10": 2.7},
    ("forest", "control"): {"r2": 0.73, "q10": 2.2},
    ("forest", "litter_removal"): {"r2": 0.74, "q10": 2.2},
}


# --------------------------------------------------------------------------- download
def fetch_hourly(lat, lon, start, end, cache_dir, refresh=False, retries=5):
    """Baixa (ou lê do cache) o clima horário de um ponto."""
    f = cache_dir / f"hourly_{lat:.4f}_{lon:.4f}_{start}_{end}.csv"
    if f.exists() and not refresh:
        print(f"  [cache] {f.name}")
        return pd.read_csv(f, parse_dates=["time"])

    params = dict(
        latitude=lat, longitude=lon, start_date=start, end_date=end,
        hourly=",".join(HOURLY), timezone=TIMEZONE,
    )
    print(f"  [Open-Meteo] lat={lat} lon={lon} {start} -> {end}")
    r = None
    for i in range(retries):
        r = requests.get(URL, params=params, timeout=120)
        if r.status_code == 200:
            break
        if 400 <= r.status_code < 500 and r.status_code != 429:
            print("  Erro da API:", r.text[:300])
            r.raise_for_status()
        wait = 5 * 2 ** i if r.status_code == 429 else 2 ** i
        print(f"  Tentativa {i + 1} falhou (status {r.status_code}); aguardando {wait}s")
        time.sleep(wait)
    else:
        r.raise_for_status()

    js = r.json()
    if "hourly" not in js:
        raise RuntimeError(f"Resposta inesperada da API: {str(js)[:300]}")
    df = pd.DataFrame(js["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(f, index=False)
    time.sleep(1)  # gentileza com a API gratuita
    return df


# --------------------------------------------------------------------------- agregações
def to_daily(h):
    """Resume o clima horário em valores diários (dia local)."""
    h = h.copy()
    h["date"] = h["time"].dt.normalize()
    g = h.groupby("date")
    daily = pd.DataFrame({
        "temp_mean": g["temperature_2m"].mean(),
        "precip_sum": g["precipitation"].sum(min_count=1),
        "et0_sum": g["et0_fao_evapotranspiration"].sum(min_count=1),
        "soil_t_mean": g["soil_temperature_0_to_7cm"].mean(),
        "soil_moist_mean": g["soil_moisture_0_to_7cm"].mean(),
    })
    daily.index = pd.DatetimeIndex(daily.index)
    return daily.sort_index()


def window_features(daily, date):
    """Características climáticas das janelas que terminam no dia anterior à coleta."""
    out = {}
    for n in WINDOWS:
        w = daily.loc[date - pd.Timedelta(days=n): date - pd.Timedelta(days=1)]
        complete = len(w) == n
        def val(x):
            return x if complete else np.nan
        out[f"temp_mean_{n}d"] = val(w["temp_mean"].mean())
        out[f"precip_sum_{n}d"] = val(w["precip_sum"].sum())
        out[f"dry_days_{n}d"] = val(int((w["precip_sum"] < DRY_DAY_MM).sum()))
        out[f"et0_sum_{n}d"] = val(w["et0_sum"].sum())
        out[f"water_balance_{n}d"] = val(w["precip_sum"].sum() - w["et0_sum"].sum())
        out[f"soil_t_mean_{n}d"] = val(w["soil_t_mean"].mean())
        out[f"soil_moist_mean_{n}d"] = val(w["soil_moist_mean"].mean())
    # valores do próprio dia da coleta (usados na validação)
    if date in daily.index:
        out["om_soil_t_day"] = daily.loc[date, "soil_t_mean"]
        out["om_soil_moist_day"] = daily.loc[date, "soil_moist_mean"]
        out["om_temp_day"] = daily.loc[date, "temp_mean"]
    else:
        out["om_soil_t_day"] = out["om_soil_moist_day"] = out["om_temp_day"] = np.nan
    return out


def fit_q10(T, y):
    """Regressão log-linear y = a * exp(b*T); Q10 = exp(10*b); R² na escala log (como no artigo)."""
    T = np.asarray(T, float)
    y = np.asarray(y, float)
    m = np.isfinite(T) & np.isfinite(y) & (y > 0)
    if m.sum() < 3:
        return dict(n=int(m.sum()), q10=np.nan, r2=np.nan)
    ly = np.log(y[m])
    b, a = np.polyfit(T[m], ly, 1)
    pred = a + b * T[m]
    ss_res = ((ly - pred) ** 2).sum()
    ss_tot = ((ly - ly.mean()) ** 2).sum()
    return dict(n=int(m.sum()), q10=float(np.exp(10 * b)), r2=float(1 - ss_res / ss_tot))


# --------------------------------------------------------------------------- principal
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="dados_solo_austria_medias.csv")
    ap.add_argument("--out", default="saida_clima")
    ap.add_argument("--refresh", action="store_true", help="ignora o cache e baixa de novo")
    args = ap.parse_args(argv)

    out = Path(args.out)
    cache = out / "cache_open_meteo"
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv, parse_dates=["date"])
    print(f"Lidas {len(df)} linhas de {args.csv}")

    # ---- 1) clima por local
    daily_by_site, feats = {}, []
    for site, g in df.groupby("site"):
        lat, lon = float(g["latitude"].iloc[0]), float(g["longitude"].iloc[0])
        start = (g["date"].min() - pd.Timedelta(days=max(WINDOWS))).strftime("%Y-%m-%d")
        end = g["date"].max().strftime("%Y-%m-%d")
        print(f"Local: {site}")
        hourly = fetch_hourly(lat, lon, start, end, cache, refresh=args.refresh)
        daily = to_daily(hourly)
        daily.to_csv(out / f"clima_diario_{site}.csv", index_label="date")
        daily_by_site[site] = daily
        for d in sorted(g["date"].unique()):
            row = window_features(daily, pd.Timestamp(d))
            row.update(site=site, date=pd.Timestamp(d))
            feats.append(row)

    feats = pd.DataFrame(feats)
    full = df.merge(feats, on=["site", "date"], how="left")
    full.to_csv(out / "dados_solo_clima.csv", index=False)
    print(f"-> {out/'dados_solo_clima.csv'}  ({full.shape[0]} linhas, {full.shape[1]} colunas)")
    miss = full.filter(regex=r"_(7|14|30|60|90)d$").isna().sum().sum()
    if miss:
        print(f"AVISO: {miss} valores de janela ficaram vazios (dados diários incompletos).")

    # ---- 2) validação A: solo medido x solo Open-Meteo (média dos tratamentos por local e data)
    sd = (full.groupby(["site", "date"])
              .agg(soil_t_meas=("soil_temp_c_mean", "mean"),
                   soil_w_meas=("soil_water_content_pct_mean", "mean"),
                   soil_t_om=("om_soil_t_day", "first"),
                   soil_m_om=("om_soil_moist_day", "first"))
              .reset_index())
    rows = []
    for site, g in sd.groupby("site"):
        g = g.dropna(subset=["soil_t_meas", "soil_t_om"])
        diff = g["soil_t_om"] - g["soil_t_meas"]
        rows.append(dict(site=site, variavel="temperatura do solo (°C)", n=len(g),
                         correlacao_pearson=g["soil_t_om"].corr(g["soil_t_meas"]),
                         vies_open_meteo_menos_medido=diff.mean(),
                         rmse=float(np.sqrt((diff ** 2).mean()))))
        g2 = sd[sd.site == site].dropna(subset=["soil_w_meas", "soil_m_om"])
        rows.append(dict(site=site, variavel="umidade do solo (ordem de postos, Spearman)", n=len(g2),
                         correlacao_pearson=g2["soil_m_om"].rank().corr(g2["soil_w_meas"].rank()),
                         vies_open_meteo_menos_medido=np.nan, rmse=np.nan))
    val = pd.DataFrame(rows)
    val.to_csv(out / "validacao_solo_open_meteo.csv", index=False)
    print("\nVALIDAÇÃO A - Open-Meteo x medidas do artigo (umidade: unidades diferentes, use só a correlação)")
    print(val.round(3).to_string(index=False))

    # ---- 3) validação B: Q10 da respiração (medido vs Open-Meteo) x Figura 4 do artigo
    rows = []
    for (site, trt), g in full.groupby(["site", "treatment"]):
        y = g["respiration_ngC_g_h_mean"].values
        for nome, T in (("temperatura medida no campo", g["soil_temp_c_mean"].values),
                        ("temperatura do solo Open-Meteo", g["om_soil_t_day"].values)):
            r = fit_q10(T, y)
            pub = PUBLISHED_Q10.get((site, trt), {})
            rows.append(dict(site=site, treatment=trt, temperatura_usada=nome, n=r["n"],
                             q10=r["q10"], r2_log=r["r2"],
                             q10_artigo_fig4=pub.get("q10"), r2_artigo_fig4=pub.get("r2")))
    q = pd.DataFrame(rows)
    q.to_csv(out / "validacao_q10_respiracao.csv", index=False)
    print("\nVALIDAÇÃO B - Q10 da respiração (médias por data; o artigo usou amostras individuais)")
    print(q.round(2).to_string(index=False))

    # ---- 4) precipitação de junho a outubro: 2018 x 2019
    pr = []
    for site, d in daily_by_site.items():
        m = d["precip_sum"].resample("MS").sum(min_count=1)
        for ts, v in m.items():
            if ts.year in (2018, 2019) and 6 <= ts.month <= 10:
                pr.append(dict(site=site, ano=ts.year, mes=ts.month, precip_mm=v))
    pr = pd.DataFrame(pr)
    if not pr.empty:
        piv = pr.pivot_table(index=["site", "mes"], columns="ano", values="precip_mm").reset_index()
        piv.to_csv(out / "precipitacao_jun_out_2018_2019.csv", index=False)
        tot = pr.groupby(["site", "ano"])["precip_mm"].sum().unstack()
        print("\nPRECIPITAÇÃO jun-out (mm) - o artigo descreve verão e outono de 2018 como secos")
        print(tot.round(0).to_string())
    print(f"\nPronto. Arquivos em: {out.resolve()}")


if __name__ == "__main__":
    sys.exit(main())