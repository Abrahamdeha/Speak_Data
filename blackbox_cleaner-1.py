"""
blackbox_cleaner.py
===================
Pipeline de nettoyage de données générique pour préparer des jeux
de transactions et d'utilisateurs à un modèle de détection de fraude.

Ce module contient une fonction unique `clean_data` permettant de charger, nettoyer,
fusionner et anonymiser deux jeux de données JSON (utilisateurs et transactions).
Il applique un nettoyage dynamique, masque les informations personnelles (PII),
formate les dates, et convertit toutes les valeurs en chaînes arrondies à deux décimales.

Aucune sortie console (print, affichage graphique) n’est utilisée.
Le résultat final est retourné sous forme de DataFrame et sauvegardé en CSV.
"""

# ==== Importation des librairies autorisées ====
from typing import Optional
from pathlib import Path
import pandas as pd
import numpy as np
import re
import json
from dateutil import parser as dateparser

# ==== 1. Fonction pour charger proprement un fichier JSON ====
def _safe_load_json(path: Path) -> pd.DataFrame:
    """Charge un fichier JSON ou JSONL (une ligne par JSON) et retourne un DataFrame."""
    text = path.read_text(encoding='utf-8')
    try:
        # On tente de charger le fichier entier comme JSON
        obj = json.loads(text)
        
        # Cas 1 : c’est une liste d’objets JSON
        if isinstance(obj, list):
            try:
                return pd.json_normalize(obj)
            except Exception:
                return pd.DataFrame(obj)
        
        # Cas 2 : c’est un seul dictionnaire JSON
        elif isinstance(obj, dict):
            try:
                return pd.json_normalize([obj])
            except Exception:
                return pd.DataFrame([obj])
    except Exception:

        # Cas 3 : format JSONL → chaque ligne est un JSON indépendant
        rows = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
        try:
            return pd.json_normalize(rows)
        except Exception:
            return pd.DataFrame(rows)

# ==== 2. Nettoyage et normalisation des noms de colonnes ====
def _normalize_colnames(df: pd.DataFrame) -> pd.DataFrame:
    """Met les noms de colonnes en minuscules, enlève les espaces et caractères spéciaux."""
    df = df.copy()
    df.columns = [
        re.sub(r"[^0-9a-zA-Z_]+", "_", str(c).strip().lower()).strip("_") 
        for c in df.columns
        ]
    return df

# ==== 3. Conversion des cellules imbriquées (dict, list) en chaînes JSON ====
def _stringify_nested_cells(df: pd.DataFrame) -> pd.DataFrame:
    """Transforme toute cellule contenant un dict/list en texte JSON."""
    df = df.copy()
    for col in df.columns:
        # If any cell is a dict or list, convert the whole column to JSON strings
        if df[col].apply(lambda x: isinstance(x, (dict, list))).any():
            df[col] = df[col].apply(
                lambda x: json.dumps(x, ensure_ascii=False) 
                if isinstance(x, (dict, list)) 
                else x)
    return df

# ==== 4. Parsing et formatage uniforme des dates ====
def _coerce_datetime(val: object) -> Optional[str]:
    """Convertit n'importe quelle date en format JJ/MM/AAAA HH:MM:SS."""
    if pd.isna(val):
        return None
    s = str(val).strip()
    if s == "":
        return None
    try:
        dt = dateparser.parse(s, fuzzy=True, dayfirst=False)
        return dt.strftime("%d/%m/%Y %H:%M:%S")
    except Exception:
        return None

# ==== 5. Masquage des emails ====
def _mask_email(email: str) -> str:
    """Masque l'email en ne masquant que la partie locale, le domaine reste visible."""
    if not email or pd.isna(email):
        return ""
    email = str(email).strip()

    if '@' not in email:
        return email[0] + "****"

    local, domain = email.split("@", 1)

    # Masquage uniquement de la partie locale
    masked_local = local[0] + "****"

    return masked_local + "@" + domain

# ==== 6. Masquage des numéros d'identité nationale ====
def _mask_national_id(nid: str) -> str:
    """Masque un identifiant en ne gardant que les 3 premiers caractères."""
    s = str(nid) if nid is not None else ""
    s = s.strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return ""
    keep = 3
    masked = s[:keep] + "X" * max(0, len(s)-keep)
    return masked


# ==== 7. Arrondir les nombres à 2 décimales et convertir en chaîne ====
def _round_numeric_and_to_str(val: object) -> str:
    """Convertit tout en chaîne, arrondit les nombres à 2 décimales."""
    if pd.isna(val):
        return ""
    try:
        # On teste si la valeur est numérique ou ressemble à un nombre
        
        if (
            isinstance(val, (int, float, np.integer, np.floating)) 
            or (
                isinstance(val, str) 
                and re.match(r"^-?\d+(?:\.\d+)?$", val.strip())
            )
        ):
            num = float(val)
            return f"{num:.2f}"
    except Exception:
        pass
    return str(val).strip()


# ==== 8. Fonction principale : clean_data ====
def clean_data(users_path: str, transactions_path: str, output_path: str) -> pd.DataFrame:
    """
    Nettoie, fusionne et anonymise deux fichiers JSON (utilisateurs et transactions).

    Étapes :
    1. Charger et normaliser les deux fichiers JSON.
    2. Nettoyer et convertir toutes les valeurs au format texte.
    3. Fusionner les jeux de données selon l'identifiant utilisateur commun.
    4. Masquer les informations personnelles (emails, noms, identifiants nationaux).
    5. Convertir les dates au format JJ/MM/AAAA HH:MM:SS.
    6. Renommer la première colonne en `ID` (anciennement `tx_id`).
    7. Sauvegarder le résultat au format CSV dans `output_path`.

    Paramètres :
        users_path (str) : chemin vers le fichier JSON des utilisateurs.
        transactions_path (str) : chemin vers le fichier JSON des transactions.
        output_path (str) : chemin de sortie pour le fichier CSV nettoyé.

    Retour :
        pd.DataFrame : le tableau nettoyé, fusionné et anonymisé.
    """

     # --- Chargement des fichiers JSON ---
    users_p = Path(users_path)
    tx_p = Path(transactions_path)
    out_p = Path(output_path)

    users_df = _safe_load_json(users_p)
    tx_df = _safe_load_json(tx_p)

    # --- Normalisation des noms de colonnes ---
    users_df = _normalize_colnames(users_df)
    tx_df = _normalize_colnames(tx_df)

    # --- Conversion des cellules imbriquées ---
    users_df = _stringify_nested_cells(users_df)
    tx_df = _stringify_nested_cells(tx_df)

    # --- Suppression des doublons ---
    users_df = users_df.drop_duplicates().reset_index(drop=True)
    tx_df = tx_df.drop_duplicates().reset_index(drop=True)

    # --- Détection dynamique des colonnes d'identifiants utilisateurs ---
    user_id_candidates = [
        c for c in users_df.columns 
        if re.search(r"user|customer|client|cust", c)
    ]
    tx_user_id_candidates = [
        c for c in tx_df.columns 
        if re.search(r"user|customer|client|cust", c)
    ]
    merge_on_users = user_id_candidates[0] if user_id_candidates else None
    merge_on_tx = tx_user_id_candidates[0] if tx_user_id_candidates else None
    if merge_on_users is None and 'id' in users_df.columns:
        merge_on_users = 'id'
    if merge_on_tx is None and 'user_id' in tx_df.columns:
        merge_on_tx = 'user_id'

    # --- Détection de la colonne tx_id ---
    tx_id_col = None
    for c in tx_df.columns:
        if re.search(r"tx[_-]?id|transaction[_-]?id|txid", c):
            tx_id_col = c
            break
    if tx_id_col and tx_id_col != 'tx_id':
        tx_df = tx_df.rename(columns={tx_id_col: 'tx_id'})
        tx_id_col = 'tx_id'

    # --- Détection et nettoyage des montants ---
    amount_candidates = [
        c for c in tx_df.columns 
        if re.search(r"amount|amt|value|total", c)
    ]
    if amount_candidates:
        amt_col = amount_candidates[0]
        tx_df[amt_col] = pd.to_numeric(tx_df[amt_col], errors='coerce')

    # --- Détection de la colonne de date ---
    ts_candidates = [
        c for c in tx_df.columns 
        if re.search(r"time|date|timestamp|tstamp", c)
        ]
    ts_col = ts_candidates[0] if ts_candidates else None

    # --- Détection de la méthode de paiement ---
    pay_candidates = [
        c for c in tx_df.columns 
        if re.search(r"payment|pay_method|method|payment_method", c)
    ]
    pay_col = pay_candidates[0] if pay_candidates else None
    if pay_col and pay_col in tx_df.columns:
        tx_df[pay_col] = (
            tx_df[pay_col]
            .astype(str)
            .str.strip()
            .str.title()
            .replace({'Nan':'', 'None':''})
        )

    # --- Détection de la colonne de fraude ---
    fraud_candidates = [
        c for c in tx_df.columns 
        if re.search(r"fraud|is_fraud|label|isfraud", c)
    ]
    fraud_col = fraud_candidates[0] if fraud_candidates else None
    
    # Normalisation du flag de fraude
    if fraud_col and fraud_col in tx_df.columns:
        def _to_flag(x):
            if pd.isna(x):
                return 0
            s = str(x).strip().lower()
            if s in {'1','true','yes','y','t'}:
                return 1
            try:
                if float(s) != 0:
                    return 1
            except Exception:
                pass
            return 0
        tx_df[fraud_col] = tx_df[fraud_col].apply(_to_flag)

    # --- Fusion des deux DataFrames ---
    if merge_on_users and merge_on_tx:
        merged = tx_df.merge(
            users_df, 
            left_on=merge_on_tx, 
            right_on=merge_on_users, 
            how='left', 
            suffixes=('', '_user'))
    else:
        if len(tx_df) == len(users_df) and len(tx_df) > 0:
            merged = pd.concat(
                [tx_df.reset_index(drop=True), users_df.reset_index(drop=True)], 
                axis=1)
        else:
            users_pref = users_df.add_prefix('user_')
            merged = pd.concat(
                [tx_df.reset_index(drop=True), users_pref.reset_index(drop=True)], 
                axis=1)

    # --- Suppression des doublons sur tx_id ---
    if 'tx_id' in merged.columns:
        merged = merged.drop_duplicates(subset=['tx_id']).reset_index(drop=True)

    # --- Masquage des PII (e-mails, identifiants) ---
    for col in merged.columns:
        col_lower = col.lower()
        if 'email' in col_lower:
            merged[col] = merged[col].apply(_mask_email)
        if re.search(r"national|nid|idnumber|id_number|ssn|social", col_lower):
            merged[col] = merged[col].apply(_mask_national_id)
        
    # --- Standardisation des dates ---
    if ts_col and ts_col in merged.columns:
        merged[ts_col] = merged[ts_col].apply(_coerce_datetime)

    # Autres colonnes "date-like"
    date_like = [
        c for c in merged.columns 
        if re.search(r"created|date|joined|dob|account_created|accountcreated", c)
    ]
    for c in date_like:
        merged[c] = merged[c].apply(_coerce_datetime)

    # --- Conversion des valeurs numériques + forçage string ---
    numeric_cols = merged.select_dtypes(include=[np.number]).columns.tolist()
    for c in numeric_cols:
        merged[c] = merged[c].apply(lambda v: _round_numeric_and_to_str(v))

    for c in merged.columns:
        if c not in numeric_cols:
            merged[c] = merged[c].apply(lambda v: _round_numeric_and_to_str(v))

    # --- Renommage de tx_id en ID ---
    if 'tx_id' in merged.columns:
        merged = merged.rename(columns={'tx_id': 'ID'})
    else:
        for c in merged.columns:
            if re.search(r"tx[_-]?id|transaction[_-]?id|txid", c):
                merged = merged.rename(columns={c: 'ID'})
                break

    # Si pas d'ID → on en crée un
    if 'ID' not in merged.columns:
        merged.insert(0, 'ID', [f"TXN{str(i+1).zfill(6)}" for i in range(len(merged))])
    else:
        cols = merged.columns.tolist()
        cols.insert(0, cols.pop(cols.index('ID')))
        merged = merged[cols]
    

    # --- Réorganisation des colonnes principales ---
    preferred_order = [
        'ID',
        'user_id',
        'amount',
        'timestamp',
        'payment_method',
        'is_fraud', 
        'first_name',
        'last_name',
        'email',
        'account_created',
        'national_id', 
        'internal_notes',
        'City',
        'Country'
    ]
    cols_final = []
    for col in preferred_order:
        if col in merged.columns and col not in cols_final:
            cols_final.append(col)
    for col in merged.columns:
        if col not in cols_final:
            cols_final.append(col)
    merged = merged[cols_final]

    # --- Conversion finale en chaînes et sauvegarde ---
    merged = merged.fillna('').astype(str)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_p, index=False)
    return merged
