#!/usr/bin/env python3

# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

"""
PDF-Report (deutsch) der TI-Kalibrierung central_europe_TI aus den Tabellen
und Grafiken von evaluate_bands.py, compare_oggm_params.py und
compare_stations.py (diese zuerst laufen lassen). Braucht reportlab (nicht in
environment.yml: pip install reportlab, oder PYTHONPATH auf eine Kopie).

Run from the repository root:
    python experiments/central_europe_TI/make_report.py
Writes experiments/central_europe_TI/report_central_europe_TI.pdf.
"""

import datetime
import os

import pandas as pd
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image as RLImage, KeepTogether, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

HERE = os.path.dirname(os.path.abspath(__file__))
TABLES = os.path.join(HERE, 'tables')
PLOTS = os.path.join(HERE, 'plots')
WIDTH = A4[0] - 3.6 * cm

styles = getSampleStyleSheet()
H1 = ParagraphStyle('h1', parent=styles['Heading1'], fontSize=15, spaceBefore=10, spaceAfter=6)
H2 = ParagraphStyle('h2', parent=styles['Heading2'], fontSize=12, spaceBefore=8, spaceAfter=4)
BODY = ParagraphStyle('body', parent=styles['BodyText'], fontSize=9.5, leading=13)
BULLET = ParagraphStyle('bullet', parent=BODY, leftIndent=12, bulletIndent=2)
CAPTION = ParagraphStyle('caption', parent=BODY, fontSize=8.5, leading=11,
                         textColor=colors.HexColor('#444444'), spaceAfter=8)
SMALL = ParagraphStyle('small', parent=BODY, fontSize=7.5, leading=9)
TITLE = ParagraphStyle('title', parent=styles['Title'], fontSize=19, alignment=TA_CENTER)


def p(text, style=BODY):
    return Paragraph(text, style)


def bullets(items):
    return [Paragraph(item, BULLET, bulletText='•') for item in items]


def figure(name, caption, width=WIDTH, max_height=19 * cm):
    path = os.path.join(PLOTS, f'{name}.png')
    w, h = Image.open(path).size
    width = min(width, max_height * w / h)
    return KeepTogether([RLImage(path, width=width, height=width * h / w), p(caption, CAPTION)])


def table(rows, widths, header_rows=1, font=8):
    t = Table(rows, colWidths=widths, repeatRows=header_rows)
    t.setStyle(TableStyle([
        ('FONT', (0, 0), (-1, -1), 'Helvetica', font),
        ('FONT', (0, 0), (-1, header_rows - 1), 'Helvetica-Bold', font),
        ('LINEABOVE', (0, 0), (-1, 0), 0.8, colors.black),
        ('LINEBELOW', (0, header_rows - 1), (-1, header_rows - 1), 0.5, colors.black),
        ('LINEBELOW', (0, -1), (-1, -1), 0.8, colors.black),
        ('ALIGN', (1, 0), (-1, -1), 'RIGHT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0, header_rows), (-1, -1), [colors.white, colors.HexColor('#f2f2f2')]),
        ('TOPPADDING', (0, 0), (-1, -1), 2), ('BOTTOMPADDING', (0, 0), (-1, -1), 2)]))
    return t


def load():
    summary = pd.read_csv(os.path.join(TABLES, 'band_summary.tsv'), sep='\t')
    scores = pd.read_csv(os.path.join(TABLES, 'band_scores.tsv'), sep='\t')
    params = pd.read_csv(os.path.join(TABLES, 'params_vs_oggm.tsv'), sep='\t')
    stations = pd.read_csv(os.path.join(TABLES, 'stations.tsv'), sep='\t')
    return summary, scores, params, stations


def numbers(summary, scores, params, stations):
    """Kennzahlen für den Text."""
    def s(quantity, variant, col):
        row = summary[(summary.quantity == quantity) & (summary.variant == variant)]
        return float(row[col].iloc[0])
    n = {'glaciers': int(s('band annual', 'TI', 'glaciers')),
         'records': int(s('band annual', 'TI', 'records'))}
    for key, quantity in [('gw', 'glacier-wide annual'), ('band', 'band annual'), ('prof', 'mean profile')]:
        for var, label in [('TI', 'ti'), ('TI calendar year', 'cal'), ('ELA', 'ela')]:
            n[f'{key}_rmse_{label}'] = s(quantity, var, 'rmse')
            n[f'{key}_r_{label}'] = s(quantity, var, 'r_median')
    n['summer_bias'] = s('band summer', 'TI', 'bias')
    n['winter_r'] = s('band winter', 'TI', 'r_median')
    gw = scores[(scores.quantity == 'glacier-wide annual')].pivot_table(index='glacier', columns='variant',
                                                                          values='r')
    n['r_better'] = int((gw['TI'] > gw['TI calendar year']).sum())
    for flow, g in params.groupby('ice_flow'):
        key = 'igm' if flow else 'nf'
        n[f'{key}_n'] = len(g)
        for q in ['melt_f', 'prcp_fac', 'temp_bias']:
            n[f'{key}_{q}_shift'] = g[f'{q}_shift'].median()
    n['melt_f_med'] = params.melt_f.median()
    n['r_mt'] = params.r_melt_f_temp_bias.median()
    n['st_t_bias'] = stations.t_bias.median()
    n['st_t_bias_jja'] = stations.t_bias_JJA.median()
    n['frost_minus_st'] = (stations.frost_temp_bias - stations.t_bias).median()
    n['frost_minus_st_jja'] = (stations.frost_temp_bias - stations.t_bias_JJA).median()
    n['oggm_minus_st'] = (stations.oggm_temp_bias - stations.t_bias).median()
    n['r_st_frost'] = stations.t_bias.corr(stations.frost_temp_bias)
    n['p_ratio'] = stations.p_ratio.median()
    n['p_ratio_winter'] = stations.p_ratio_winter.median()
    n['t_r'] = stations.t_r_anom.median()
    n['p_r'] = stations.p_r_anom.median()
    n['p_z'] = stations.p_station_z.median()
    n['worse'] = [r.glacier for r in params.itertuples() if r.misfit_posterior > r.misfit_prior]
    return n


def glacier_table(scores, params, stations):
    gw = scores[scores.quantity == 'glacier-wide annual'].pivot_table(
        index='rgi_id', columns='variant', values=['r', 'rmse', 'n'])
    source = scores.drop_duplicates('rgi_id').set_index('rgi_id').source
    st = stations.set_index('rgi_id')
    rows = [['Gletscher', 'Fluss', 'Daten', 'Jahre', 'r TI', 'RMSE\nTI', 'RMSE\nELA',
             'melt_f\nOGGM > FROST', 'prcp_fac\nOGGM > FROST', 'temp_bias\nOGGM > FROST', 'dT\nStation']]
    for r in params.sort_values(['ice_flow', 'glacier'], ascending=[False, True]).itertuples():
        if r.rgi_id not in gw.index:
            continue
        g = gw.loc[r.rgi_id]
        rows.append([r.glacier[:24], 'IGM' if r.ice_flow else '-', source[r.rgi_id],
                     f"{g[('n', 'TI')]:.0f}", f"{g[('r', 'TI')]:.2f}", f"{g[('rmse', 'TI')]:.2f}",
                     f"{g[('rmse', 'ELA')]:.2f}",
                     f'{r.oggm_melt_f:.1f} > {r.melt_f:.1f}', f'{r.oggm_prcp_fac:.2f} > {r.prcp_fac:.2f}',
                     f'{r.oggm_temp_bias:+.1f} > {r.temp_bias:+.1f}', f'{st.t_bias[r.rgi_id]:+.1f}'])
    return table(rows, [3.6 * cm, 0.9 * cm, 1.3 * cm, 0.9 * cm, 0.9 * cm, 1.0 * cm, 1.0 * cm,
                        2.1 * cm, 2.1 * cm, 2.1 * cm, 1.2 * cm], font=7)


def story(summary, scores, params, stations):
    n = numbers(summary, scores, params, stations)
    today = datetime.date.today().strftime('%d.%m.%Y')
    out = [p('TI-Kalibrierung Mitteleuropa (central_europe_TI)', TITLE),
           p(f'Oskar Herrmann, Stand {today}', ParagraphStyle('c', parent=BODY, alignment=TA_CENTER)),
           Spacer(1, 10)]

    out += [p('Zusammenfassung', H1)]
    out += bullets([
        f'Das OGGM-Temperaturindex-Modell (TI) wurde für {n["glaciers"]} Gletscher mit FROST (ES-MDA) '
        f'gegen die Hugonnet-Höhenänderung 2000-2020 je Höhenband kalibriert '
        f'({n["igm_n"]} mit Eisfluss in IGM, {n["nf_n"]} ohne Eisfluss) und gegen '
        f'{n["records"]} GLAMOS/WGMS-Band-Jahre der Massenbilanz evaluiert.',
        f'Die jährliche gletscherweite Massenbilanz wird gut getroffen: RMSE {n["gw_rmse_ti"]:.2f} m w.e. '
        f'(ELA-Modell {n["gw_rmse_ela"]:.2f}), Korrelation zwischen den Jahren r = {n["gw_r_ti"]:.2f} '
        f'(Median). Der Gewinn gegenüber dem ELA-Modell ist die Jahr-zu-Jahr-Variabilität; das mittlere '
        f'Profil ist ähnlich gut (RMSE {n["prof_rmse_ti"]:.2f} vs. {n["prof_rmse_ela"]:.2f}).',
        f'Der Vergleich über die tatsächlichen Messperioden statt Kalenderjahre erhöht r auf '
        f'{n["r_better"]} von {n["glaciers"]} Gletschern.',
        f'Schwächen: Sommerschmelze zu gering (Bias +{n["summer_bias"]:.2f} m w.e. je Band), '
        f'Winterbilanz schwach korreliert (r = {n["winter_r"]:.2f}), gekrümmtes Bandprofil.',
        f'Kalibrierte Parameter gegenüber OGGM: melt_f niedriger, prcp_fac niedriger, temp_bias höher. '
        f'melt_f und temp_bias sind stark antikorreliert (r = {n["r_mt"]:.2f}): die Daten bestimmen vor '
        f'allem ihre Kombination.',
        f'Stationsvergleich (MeteoSchweiz, GeoSphere Austria): W5E5 ist auf Stationshöhe im Median '
        f'{n["st_t_bias"]:.1f} °C zu kalt (Sommer {n["st_t_bias_jja"]:.1f} °C). OGGMs temp_bias liegt '
        f'nahe daran (+{n["oggm_minus_st"]:.1f} °C), FROSTs temp_bias ca. {n["frost_minus_st"]:.1f} °C darüber. '
        f'Zusammen mit dem niedrigen melt_f (Median {n["melt_f_med"]:.1f}) deutet das auf Kompensation '
        f'statt Physik hin.'])

    out += [p('1  Aufbau', H1)]
    out += bullets([
        'SMB: OGGM-TI-Modell (melt_f, prcp_fac, temp_bias) auf dem monatlichen GSWP3-W5E5-Klima der '
        'OGGM-Shop-Daten (climate_historical.nc: eine 0,5°-Zelle, Temperatur auf der Zellhöhe, '
        '-6,5 K/km). Priori-Mittel aus OGGMs eigener Kalibrierung (mb_calib.json), Standardabweichung '
        '2,5 / 0,5 / 1,0.',
        'Beobachtung: Hugonnet et al. (2021) dh/dt 2000-01-01 bis 2020-01-01 je 50-m-Höhenband.',
        f'{n["igm_n"]} Gletscher mit Eisfluss (IGM, eingefrorene Inversion, Emulator in den '
        f'Vorwärtsläufen neu trainiert, 32 Member); {n["nf_n"]} ohne Eisfluss (dh/dt = SMB mit '
        'Ausschmelzen, 128 Member). Jeweils 4 ES-MDA-Iterationen.',
        'Evaluation (evaluate_bands.py): jeder GLAMOS/WGMS-Bandwert mit seinen Daten (GLAMOS: '
        'schwankende Feldbegehungen; WGMS: meist 1. Okt - 30. Apr - 30. Sep). Das TI-Modell jedes '
        'Posterior-Members läuft auf der mittleren Bandhöhe genau über diese Periode; so werden auch '
        'Winter- und Sommerbilanz verglichen. Referenzen: dasselbe Modell im Kalenderjahr (wie in der '
        'Kalibrierung) und die konstante SMB der ELA-Kalibrierung.'])

    out += [p('2  Evaluation gegen GLAMOS / WGMS', H1)]
    rows = [['', 'TI\nMessperioden', 'TI\nKalenderjahr', 'ELA']]
    for label, key, unit in [('Gletscherweite Jahresbilanz: RMSE', 'gw_rmse', 'm w.e.'),
                             ('Gletscherweite Jahresbilanz: r (Median)', 'gw_r', ''),
                             ('Band-Jahr: RMSE', 'band_rmse', 'm w.e.'),
                             ('Band-Jahr: r (Median)', 'band_r', ''),
                             ('Mittleres Profil 2000-2019: RMSE', 'prof_rmse', 'm w.e./a')]:
        rows.append([f'{label} {"(" + unit + ")" if unit else ""}'] +
                    [f'{n[f"{key}_{v}"]:.2f}' for v in ['ti', 'cal', 'ela']])
    out += [table(rows, [8.5 * cm, 2.6 * cm, 2.6 * cm, 2.0 * cm], font=8.5),
            p(f'Mittel der Werte je Gletscher über {n["glaciers"]} Gletscher (tables/band_summary.tsv). '
              'Die r-Werte des ELA-Modells sind bedeutungslos klein, da seine SMB konstant ist.', CAPTION)]
    out += [figure('band_timeseries', 'Abb. 1: Gletscherweite Jahresbilanz je Messjahr: gemessen '
                   '(schwarz), TI über die Messperiode (rot), TI im Kalenderjahr (orange gestrichelt), '
                   'ELA (grau).')]
    out += [p('Auffällige Gletscher', H2)]
    out += bullets([
        'Basòdino: Modell in jedem Jahr 1-1,5 m w.e. zu negativ. Kalibriert auf Hugonnet; geodätische '
        'und glaziologische Bilanz widersprechen sich hier (wie im ELA-Lauf).',
        'Zettalunitzkees, Vedretta Lunga: Modell zu negativ; Hintereisferner ab 2005 zu positiv; '
        'Plaine Morte verfehlt 2015-2019.',
        'Rhone, Vadret Pers, Findel (IGM): Jahresgang getroffen, aber teils deutlicher Offset.'])
    out += [figure('band_scatter', 'Abb. 2: Gemessene gegen modellierte Bandwerte (alle Band-Jahre '
                   'zusammen; daher weichen die Zahlen in den Titeln von den Mittelwerten je Gletscher '
                   'ab). Jahresbilanz TI, TI im Kalenderjahr, ELA; Winter- und Sommerbilanz TI.')]
    out += bullets([
        'Die Jahreswolke ist gekrümmt: oben (gemessen nahe 0) zu positiv, an der Zunge weniger negativ '
        'als gemessen. Wahrscheinliche Ursachen: eine Klimazelle, kein Niederschlagsgradient, feste '
        'Temperaturabnahme mit der Höhe.',
        f'Sommerbilanz eng korreliert, aber zu schwach (Bias +{n["summer_bias"]:.2f}); Winterbilanz eine '
        f'lose Wolke (r = {n["winter_r"]:.2f}).'])

    out += [PageBreak(), p('3  Kalibrierte Parameter gegenüber OGGM', H1)]
    out += [figure('params_vs_oggm', 'Abb. 3: FROST-Posterior (Mittel +- Standardabweichung) gegen '
                   'OGGMs Kalibrierung (= Priori-Mittel). Grau: +-1 und +-2 Priori-Standardabweichungen. '
                   'Rechts: Korrelation der Parameter zwischen den Membern je Gletscher. Rot ohne, blau mit '
                   'Eisfluss.')]
    rows = [['Verschiebung (Priori-Std., Median)', 'melt_f', 'prcp_fac', 'temp_bias']]
    for key, label in [('nf', f'ohne Eisfluss ({n["nf_n"]})'), ('igm', f'mit Eisfluss, IGM ({n["igm_n"]})')]:
        rows.append([label] + [f'{n[f"{key}_{q}_shift"]:+.2f}' for q in ['melt_f', 'prcp_fac', 'temp_bias']])
    out += [table(rows, [7 * cm, 2.6 * cm, 2.6 * cm, 2.6 * cm], font=8.5), Spacer(1, 8)]
    out += [p('Mögliche Gründe für die Unterschiede', H2)]
    out += bullets([
        '<b>OGGM wählt einen Punkt auf einem Grat.</b> OGGM passt nur eine Zahl an (gletscherweite '
        'Bilanz) und löst die Mehrdeutigkeit mit einer Regel: prcp_fac aus dem Klima, melt_f = 5, dann '
        'temp_bias. Daher liegen viele OGGM-melt_f genau bei 5,0. FROST passt das Bandprofil an und '
        'bewegt sich entlang dieses Grats.',
        '<b>Ohne Eisfluss ist dh/dt nicht die SMB.</b> Eisfluss senkt die Oberfläche oben und hebt sie an '
        'der Zunge; das dh/dt-Profil ist flacher als das SMB-Profil. Die IGM-Gletscher zeigen aber '
        'dieselbe Richtung (melt_f und prcp_fac niedriger), mit kleinerer temp_bias-Verschiebung. Der '
        'fehlende Eisfluss erklärt die Verschiebung also höchstens teilweise.',
        '<b>Kleinere Effekte:</b> FROST vergleicht dh/dt mit der SMB in Eisäquivalent (910 kg/m³), '
        'OGGMs Ziel nutzt Hugonnets 850 kg/m³ (ca. 7 % negativer bei FROST); FROST lässt den Gletscher '
        'schrumpfen, OGGM hält die Geometrie fest.'])
    if n['worse']:
        out += [p(f'<b>Hinweis:</b> Bei {", ".join(n["worse"])} ist der Misfit nach der Kalibrierung '
                  'größer als vorher (ES-MDA mit Emulator). Diese Läufe sollten geprüft werden.', BODY)]

    out += [PageBreak(), p('4  Vergleich des Kalibrierklimas mit Wetterstationen', H1)]
    out += [p('Monatswerte 2000-2019 von MeteoSchweiz (SwissMetNet, Open Data) und GeoSphere Austria '
              '(klima-v2-1m). Je Gletscher die nächste hohe Station mit Temperatur und die nächste mit '
              'Niederschlag (Gipfelstationen wie Jungfraujoch oder Gornergrat haben keinen '
              'Niederschlagsmesser). W5E5-Temperatur mit -6,5 K/km auf Stationshöhe verschoben. Für die '
              'italienischen Gletscher Careser und Vedretta Lunga gibt es nur Sta. Maria (Val Müstair, '
              '21-27 km, anderes Tal).')]
    out += [figure('stations', 'Abb. 4: Links: Temperaturdifferenz Station - W5E5 gegen den kalibrierten '
                   'temp_bias (OGGM, FROST). Mitte: Temperaturdifferenz je Jahreszeit (grau je Gletscher, '
                   'schwarz Median). Rechts davon: Niederschlagsverhältnis Station / W5E5 gegen prcp_fac; '
                   'Korrelation der monatlichen Anomalien.')]
    out += [p('Sind die FROST-Werte plausibel?', H2)]
    out += bullets([
        f'<b>temp_bias: vergleichbar.</b> Die Stationen zeigen W5E5 im Median {n["st_t_bias"]:.1f} °C zu '
        f'kalt (Sommer {n["st_t_bias_jja"]:.1f} °C); ein positiver temp_bias ist also real, und der '
        f'kalibrierte Wert folgt den Stationen von Gletscher zu Gletscher (r = {n["r_st_frost"]:.2f}). '
        f'OGGM liegt nahe an den Stationen (+{n["oggm_minus_st"]:.1f} °C), FROST '
        f'{n["frost_minus_st"]:.1f} °C darüber ({n["frost_minus_st_jja"]:.1f} °C gegenüber dem Sommer). '
        'Über Gletschern ist die Luft im Sommer 1-2 °C kühler als an Stationen; der physikalisch '
        'richtige Wert läge also eher unter dem Stationswert. FROSTs temp_bias ist vermutlich zu hoch.',
        f'<b>prcp_fac: nur grob vergleichbar.</b> Die Messer sehen etwa so viel wie W5E5 (Verhältnis '
        f'{n["p_ratio"]:.2f}, Winter {n["p_ratio_winter"]:.2f}), stehen aber im Median auf '
        f'{n["p_z"]:.0f} m, ca. 1000 m unter den Gletschern, und fangen 20-40 % zu wenig Schnee. Mit '
        'beiden Korrekturen (x1,2-1,4 und 5-10 % mehr Niederschlag je 100 m) ergibt sich ein Faktor von '
        'etwa 2-3: die FROST-Werte sind plausibel.',
        f'<b>melt_f: kein Stationsäquivalent.</b> Literaturwerte (Hock 2003): 3-5 (Schnee) und 6-8 '
        f'(Eis) mm w.e./°C/Tag. Der FROST-Median {n["melt_f_med"]:.1f} liegt am unteren Rand.',
        f'<b>Jahr-zu-Jahr:</b> W5E5 folgt den Stationen gut (r = {n["t_r"]:.2f} Temperatur, '
        f'{n["p_r"]:.2f} Niederschlag). Die schwache Winterkorrelation der Evaluation kommt daher eher '
        'von lokalen Schneeprozessen (Wind, Lawinen) als von W5E5.',
        '<b>Fazit:</b> Einzeln liegen die FROST-Werte im plausiblen Bereich; die Kombination (temp_bias '
        'zu warm, melt_f niedrig) sieht nach Kompensation aus. Für Projektionen ist das relevant: die '
        'Reaktion auf Erwärmung hängt an melt_f und könnte zu schwach sein.'])
    out += [PageBreak(),
            figure('stations_temperature_annual', 'Abb. 5: Jahresmitteltemperatur je Gletscher: Station '
                   '(schwarz) und W5E5 auf Stationshöhe (rot). Jahre mit fehlenden Monaten fehlen.',
                   max_height=21 * cm),
            figure('stations_temperature_seasonal', 'Abb. 6: Mittlerer Jahresgang der Temperatur '
                   '2000-2019.', max_height=21 * cm),
            figure('stations_precipitation_annual', 'Abb. 7: Jahresniederschlag je Gletscher: Station '
                   '(schwarz) und W5E5 (rot).', max_height=21 * cm),
            figure('stations_precipitation_seasonal', 'Abb. 8: Mittlerer Jahresgang des Niederschlags '
                   '2000-2019. Auf den österreichischen Gletschern zeigen die Stationen ein starkes '
                   'Sommermaximum, das W5E5 unterschätzt; im Wallis ist W5E5 nasser als die Talstationen.',
                   max_height=21 * cm)]

    out += [PageBreak(), p('5  Nächste Schritte', H1)]
    out += bullets([
        'Test mit festem temp_bias (Stationswert, abzüglich Gletscherkühlung): nur melt_f und prcp_fac '
        'kalibrieren und prüfen, wie stark sich der Fit verschlechtert.',
        'Niederschlagsgradient mit der Höhe einführen (gekrümmtes Bandprofil, Winterbilanz).',
        'Split-Sample-Test: Kalibrierung auf Hugonnet 2000-2010, Evaluation 2010-2019, als Test der '
        'Übertragbarkeit für Projektionen.',
        'IGM-Läufe mit größerem Misfit nach der Kalibrierung prüfen.',
        'Projektionen: ISIMIP3b (CMIP6-SSPs, auf W5E5 biaskorrigiert) ergibt einen glatten Übergang; '
        'alps_TI_projections nutzt bisher CORDEX-RCPs, auf W5E5 2000-2019 korrigiert.'])
    out += [p('Anhang: Werte je Gletscher', H1),
            p('r und RMSE (m w.e.) der gletscherweiten Jahresbilanz über die Messperioden; Parameter OGGM > '
              'FROST-Posterior-Mittel; dT = Station - W5E5 (°C) auf Stationshöhe.', CAPTION),
            glacier_table(scores, params, stations),
            Spacer(1, 10),
            p('Skripte (Repository-Wurzel): experiments/central_europe_TI/evaluate_bands.py, '
              'compare_oggm_params.py, compare_stations.py, make_report.py. Tabellen in '
              'experiments/central_europe_TI/tables, Grafiken in experiments/central_europe_TI/plots.',
              SMALL)]
    return out


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont('Helvetica', 7.5)
    canvas.drawString(1.8 * cm, 1.0 * cm, 'FROST central_europe_TI')
    canvas.drawRightString(A4[0] - 1.8 * cm, 1.0 * cm, f'Seite {doc.page}')
    canvas.restoreState()


def main():
    path = os.path.join(HERE, 'report_central_europe_TI.pdf')
    doc = SimpleDocTemplate(path, pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm,
                            topMargin=1.6 * cm, bottomMargin=1.6 * cm,
                            title='TI-Kalibrierung Mitteleuropa', author='Oskar Herrmann')
    doc.build(story(*load()), onFirstPage=footer, onLaterPages=footer)
    print(path)


if __name__ == '__main__':
    main()
