"""Génère docs/racines-ia.svg : les modèles d'IA représentés comme des racines.

Profondeur de la racine = nombre de paramètres, sur une échelle logarithmique
(chaque graduation = ×10). Les tailles sont celles publiées par les éditeurs ;
Claude, GPT (versions récentes) et Gemini n'ont pas de taille publiée : leurs
racines sont en pointillés et leur position est seulement indicative.

Usage : python docs/racines_ia.py   (puis ouvrir le SVG, ou le convertir en PNG)
Pour ajouter un modèle, ajoute une ligne dans MODELES.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

W, H = 1920, 1200
SOL = 250                 # ligne de surface (y)
DECADE = 100              # pixels par puissance de 10
EXP0 = 5.5                # exposant affiché au niveau de la surface
TRONC_X = 960

ENCRE, ENCRE_2, ENCRE_3 = "#0b0b0b", "#52514e", "#898781"
SURFACE, TERRE_HAUT, TERRE_BAS = "#fcfcfb", "#f1e8da", "#e4d4bd"
RACINE, MIRADOR = "#6b4a2e", "#e05a22"   # palette validée (dataviz validate_palette.js)
POLICE = "Inter, 'DejaVu Sans', system-ui, sans-serif"

# (nom, ligne d'info, paramètres, x de la pointe, côté de l'étiquette)
MODELES = [
    ("BERT-base", "Google · 2018 · 110 M", 110e6, 400, "gauche"),
    ("GPT-2", "OpenAI · 2019 · 1,5 Md", 1.5e9, 1640, "droite"),
    ("Mistral 7B", "Mistral AI · 2023 · 7,3 Md", 7.3e9, 560, "gauche"),
    ("Llama 3 70B", "Meta · 2024 · 70 Md", 70e9, 1440, "droite"),
    ("GPT-3", "OpenAI · 2020 · 175 Md", 175e9, 740, "gauche"),
    ("Llama 3.1 405B", "Meta · 2024 · 405 Md", 405e9, 1190, "droite"),
]
# Disposition : de chaque côté, les racines peu profondes s'étalent loin et les profondes
# restent près du centre (comme un vrai système racinaire) : aucune étiquette n'est traversée.
# Ton modèle : mets à jour ces valeurs à chaque nouvelle version entraînée.
MIRADOR_SI = {
    "nom": "Mirador SI v0.4 · Vigie",
    "params": 10.68e6,
    "x": 1290,
    "lignes": ["10,7 M de paramètres · {taille} compressé", "créé de zéro · s'entraîne et tourne sur un simple PC"],
    "taille": "9,9 Mo",
    "precedent": ("v0.3", 4.26e6),   # ancienne version, repérée sur la même racine
}
POINTE = [("Claude", 820), ("GPT", 950), ("Gemini", 1080)]   # taille non publiée
POINTE_EXP = 12.3         # sert seulement à doser l'épaisseur des racines
HORS_ECHELLE = SOL + (12.0 - EXP0) * DECADE + 32   # début de la zone « taille non publiée »


def y_de(params: float) -> float:
    return SOL + (math.log10(params) - EXP0) * DECADE


def esc(t: str) -> str:
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def bezier(p0, p1, p2, p3, t):
    u = 1 - t
    return (u ** 3 * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t ** 3 * p3[0],
            u ** 3 * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t ** 3 * p3[1])


def controles(tip):
    """Les racines peu profondes s'écartent tôt, les profondes plongent d'abord."""
    sx = TRONC_X + max(-22, min(22, (tip[0] - TRONC_X) * 0.05))
    p0 = (sx, SOL + 2)
    dx, dy = tip[0] - p0[0], tip[1] - p0[1]
    f = max(0.0, min(1.0, (tip[1] - SOL) / (HORS_ECHELLE - SOL)))
    return (p0, (p0[0] + dx * (0.42 - 0.32 * f), p0[1] + dy * (0.2 + 0.35 * f)),
            (tip[0] - dx * (0.32 - 0.12 * f), tip[1] - dy * 0.3), tip)


def racine(tip, largeur, couleur, rng) -> str:
    """Racine effilée (forme pleine) + radicelles."""
    pts = controles(tip)
    n = 60
    gauche, droite = [], []
    phase = rng.uniform(0, math.tau)
    for i in range(n + 1):
        t = i / n
        x, y = bezier(*pts, t)
        x2, y2 = bezier(*pts, min(1, t + 1e-3))
        x1, y1 = bezier(*pts, max(0, t - 1e-3))
        tx, ty = x2 - x1, y2 - y1
        norm = math.hypot(tx, ty) or 1
        nx, ny = -ty / norm, tx / norm
        ondulation = math.sin(t * 9 + phase) * 3.0 * t      # racine un peu sinueuse
        x, y = x + nx * ondulation, y + ny * ondulation
        w = largeur * (1 - t) ** 0.85 / 2 + 0.9
        gauche.append((x + nx * w, y + ny * w))
        droite.append((x - nx * w, y - ny * w))
    contour = gauche + droite[::-1]
    d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in contour) + " Z"
    out = [f'<path d="{d}" fill="{couleur}"/>']
    for t in (0.32, 0.55, 0.74):                              # radicelles
        x, y = bezier(*pts, t)
        x2, y2 = bezier(*pts, t + 0.01)
        ang = math.atan2(y2 - y, x2 - x) + rng.choice([-1, 1]) * rng.uniform(0.6, 1.0)
        long_ = rng.uniform(26, 26 + largeur * 4)
        ex, ey = x + math.cos(ang) * long_, y + math.sin(ang) * long_
        cx, cy = x + math.cos(ang - 0.4) * long_ * 0.5, y + math.sin(ang - 0.4) * long_ * 0.5
        out.append(f'<path d="M{x:.1f},{y:.1f} Q{cx:.1f},{cy:.1f} {ex:.1f},{ey:.1f}" fill="none" '
                   f'stroke="{couleur}" stroke-opacity="0.55" stroke-width="1.6" stroke-linecap="round"/>')
    return "\n".join(out)


def texte(x, y, contenu, taille=17, couleur=ENCRE_2, poids=400, ancre="start", halo=TERRE_HAUT, extra=""):
    return (f'<text x="{x:.0f}" y="{y:.0f}" font-size="{taille}" font-weight="{poids}" fill="{couleur}" '
            f'text-anchor="{ancre}" paint-order="stroke" stroke="{halo}" stroke-width="5" '
            f'stroke-linejoin="round" {extra}>{esc(contenu)}</text>')


def marqueur(x, y, couleur):
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="9" fill="{couleur}" stroke="{TERRE_HAUT}" stroke-width="2.5"/>'


def construire() -> str:
    rng = random.Random(7)
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
         f'font-family="{POLICE}">',
         '<defs>',
         f'<linearGradient id="terre" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{TERRE_HAUT}"/>'
         f'<stop offset="1" stop-color="{TERRE_BAS}"/></linearGradient>',
         f'<linearGradient id="fondu" gradientUnits="userSpaceOnUse" x1="0" y1="{SOL}" x2="0" y2="{H - 120}">'
         f'<stop offset="0" stop-color="{RACINE}" stop-opacity="0.85"/>'
         f'<stop offset="1" stop-color="{RACINE}" stop-opacity="0.12"/></linearGradient>',
         '</defs>',
         f'<rect width="{W}" height="{SOL}" fill="{SURFACE}"/>',
         f'<rect y="{SOL}" width="{W}" height="{H - SOL}" fill="url(#terre)"/>']

    # --- titre, légende
    s.append(texte(90, 78, "Les racines de la puissance", 46, ENCRE, 700, halo=SURFACE))
    s.append(texte(90, 120, "Chaque racine est un modèle d'IA.", 21, ENCRE_2, halo=SURFACE))
    s.append(texte(90, 150, "Plus elle plonge, plus le modèle a de paramètres.", 21, ENCRE_2, halo=SURFACE))
    lx, ly = 1450, 70
    s.append(f'<circle cx="{lx}" cy="{ly}" r="8" fill="{MIRADOR}"/>')
    s.append(texte(lx + 20, ly + 6, "Ton modèle : Mirador SI", 18, ENCRE, 600, halo=SURFACE))
    s.append(f'<circle cx="{lx}" cy="{ly + 34}" r="8" fill="{RACINE}"/>')
    s.append(texte(lx + 20, ly + 40, "Taille publiée par l'éditeur", 18, ENCRE_2, halo=SURFACE))
    s.append(f'<line x1="{lx - 9}" y1="{ly + 68}" x2="{lx + 9}" y2="{ly + 68}" stroke="{RACINE}" '
             f'stroke-width="4" stroke-dasharray="5 4"/>')
    s.append(texte(lx + 20, ly + 74, "Taille non publiée (hors échelle)", 18, ENCRE_2, halo=SURFACE))

    # --- graduations (échelle logarithmique)
    s.append(texte(150, SOL + 30, "paramètres", 14, ENCRE_3, 600, "end", extra='letter-spacing="0.5"'))
    for e, lab in [(6, "1 M"), (7, "10 M"), (8, "100 M"), (9, "1 Md"), (10, "10 Md"), (11, "100 Md"),
                   (12, "1 000 Md")]:
        y = SOL + (e - EXP0) * DECADE
        s.append(f'<line x1="165" y1="{y:.0f}" x2="{W - 60}" y2="{y:.0f}" stroke="{ENCRE_2}" '
                 f'stroke-opacity="0.13" stroke-width="1" stroke-dasharray="2 6"/>')
        s.append(texte(150, y + 6, lab, 16, ENCRE_3, 500, "end", extra='font-variant-numeric="tabular-nums"'))

    # --- zone hors échelle : on ne place PAS ces modèles sur une graduation
    s.append(f'<line x1="165" y1="{HORS_ECHELLE:.0f}" x2="{W - 60}" y2="{HORS_ECHELLE:.0f}" stroke="{ENCRE_2}" '
             f'stroke-opacity="0.35" stroke-width="1.2" stroke-dasharray="8 6"/>')
    s.append(f'<path d="M142,{HORS_ECHELLE - 8:.0f} l14,-8 M142,{HORS_ECHELLE + 2:.0f} l14,-8" '
             f'stroke="{ENCRE_3}" stroke-width="1.6"/>')
    s.append(texte(150, HORS_ECHELLE + 26, "hors échelle", 14, ENCRE_3, 600, "end"))
    s.append(texte(150, HORS_ECHELLE + 44, "taille inconnue", 14, ENCRE_3, 400, "end"))

    # --- racines à taille non publiée (pointillés qui se perdent dans la profondeur)
    y_pointe = HORS_ECHELLE + 52
    for _, x in POINTE:
        p0, p1, p2, p3 = controles((x, y_pointe + 70))
        s.append(f'<path d="M{p0[0]:.0f},{p0[1]:.0f} C{p1[0]:.0f},{p1[1]:.0f} {p2[0]:.0f},{p2[1]:.0f} '
                 f'{p3[0]:.0f},{p3[1]:.0f}" fill="none" stroke="url(#fondu)" stroke-width="5" '
                 f'stroke-linecap="round" stroke-dasharray="11 9"/>')
        mx, my = bezier(p0, p1, p2, p3, 0.9)
        s.append(f'<circle cx="{mx:.0f}" cy="{my:.0f}" r="13" fill="{TERRE_BAS}" stroke="{RACINE}" '
                 f'stroke-width="2" stroke-dasharray="4 3"/>')
        s.append(texte(mx, my + 6, "?", 17, ENCRE_2, 700, "middle", halo=TERRE_BAS))

    # --- racines à taille publiée
    for nom, info, params, x, cote in MODELES:
        y = y_de(params)
        profondeur = (math.log10(params) - EXP0) / (POINTE_EXP - EXP0)
        s.append(racine((x, y), 6 + 12 * profondeur, RACINE, rng))
    for nom, info, params, x, cote in MODELES:
        y = y_de(params)
        s.append(marqueur(x, y, RACINE))
        dx, ancre = (-20, "end") if cote == "gauche" else (20, "start")
        s.append(texte(x + dx, y - 2, nom, 22, ENCRE, 700, ancre))
        s.append(texte(x + dx, y + 22, info, 17, ENCRE_2, 400, ancre))

    # --- Mirador SI : la racine mise en valeur
    nom, params, x = MIRADOR_SI["nom"], MIRADOR_SI["params"], MIRADOR_SI["x"]
    y = y_de(params)
    s.append(racine((x, y), 8, MIRADOR, rng))
    if MIRADOR_SI.get("precedent"):  # l'ancienne version : un repère sur la même racine
        label_prec, params_prec = MIRADOR_SI["precedent"]
        y_prec = y_de(params_prec)
        pts = controles((x, y))
        t = min((i / 400 for i in range(401)), key=lambda t: abs(bezier(*pts, t)[1] - y_prec))
        px, py = bezier(*pts, t)
        s.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="6" fill="{TERRE_HAUT}" stroke="{MIRADOR}" stroke-width="2.5"/>')
        s.append(texte(px - 12, py - 10, label_prec, 14, ENCRE_2, 600, "end"))
    s.append(marqueur(x, y, MIRADOR))
    s.append(f'<rect x="{x + 20}" y="{y - 46}" width="122" height="24" rx="12" fill="{MIRADOR}"/>')
    s.append(f'<text x="{x + 81}" y="{y - 29}" font-size="13" font-weight="700" fill="#ffffff" '
             f'text-anchor="middle" letter-spacing="1">TON MODÈLE</text>')
    s.append(texte(x + 20, y + 8, nom, 24, ENCRE, 700))
    for i, ligne in enumerate(MIRADOR_SI["lignes"]):
        s.append(texte(x + 20, y + 33 + 23 * i, ligne.format(taille=MIRADOR_SI["taille"]), 17, ENCRE_2))

    # --- arbre (surface)
    s.append(f'<path d="M{TRONC_X - 17},{SOL + 3} C{TRONC_X - 10},{SOL - 30} {TRONC_X - 9},{SOL - 55} '
             f'{TRONC_X - 7},{SOL - 78} L{TRONC_X + 7},{SOL - 78} C{TRONC_X + 9},{SOL - 55} {TRONC_X + 10},'
             f'{SOL - 30} {TRONC_X + 17},{SOL + 3} Z" fill="{RACINE}"/>')
    for cx, cy, r, c in [(-38, -98, 34, "#86ad84"), (36, -100, 36, "#7aa37a"), (0, -128, 44, "#8fb88c"),
                         (-6, -88, 36, "#73a074"), (24, -140, 28, "#9cc298")]:
        s.append(f'<circle cx="{TRONC_X + cx}" cy="{SOL + cy}" r="{r}" fill="{c}"/>')
    s.append(texte(TRONC_X + 92, SOL - 108, "Intelligence", 18, ENCRE_2, 600, halo=SURFACE))
    s.append(texte(TRONC_X + 92, SOL - 86, "artificielle", 18, ENCRE_2, 600, halo=SURFACE))
    s.append(f'<line x1="0" y1="{SOL}" x2="{W}" y2="{SOL}" stroke="#9a8b6a" stroke-width="2"/>')
    for gx in range(30, W, 46):  # touffes d'herbe
        h = 6 + (gx * 7) % 7
        s.append(f'<path d="M{gx},{SOL} l-4,-{h} M{gx},{SOL} l1,-{h + 3} M{gx},{SOL} l5,-{h}" '
                 f'stroke="#8aa35f" stroke-width="1.6" stroke-linecap="round"/>')

    # --- modèles de pointe : légende du bloc + notes
    s.append(texte(950, y_pointe + 108, "Claude · GPT · Gemini", 24, ENCRE, 700, "middle", halo=TERRE_BAS))
    s.append(texte(950, y_pointe + 134, "Modèles de pointe : taille non publiée par Anthropic, OpenAI et Google",
                   17, ENCRE_2, 400, "middle", halo=TERRE_BAS))
    s.append(texte(90, H - 52, "Profondeur = nombre de paramètres, échelle logarithmique (chaque graduation = ×10). "
                   "Tailles publiées par les éditeurs ; Mirador SI mesuré dans ce projet.", 15, ENCRE_2,
                   halo=TERRE_BAS))
    s.append(texte(90, H - 28, "La taille ne fait pas toute la puissance : les données, l'entraînement et "
                   "l'architecture comptent aussi.", 15, ENCRE_2, halo=TERRE_BAS))
    s.append("</svg>")
    return "\n".join(s)


if __name__ == "__main__":
    out = Path(__file__).with_name("racines-ia.svg")
    out.write_text(construire(), encoding="utf-8")
    print(f"Image écrite : {out}")
