import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.stats import chi2_contingency


def _prep_cat(s: pd.Series, na_token: str) -> pd.Series:
    # Evita FutureWarning: primero object, luego NA→token sin downcasting silencioso, luego category
    return s.astype("object").where(s.notna(), na_token).astype("category")


def _cramers_v_bias_corrected(x: pd.Series, y: pd.Series, *, na_token: str) -> float:
    """Cramér's V corregido (Bergsma 2013) con crosstab eficiente."""
    x_ = _prep_cat(x, na_token)
    y_ = _prep_cat(y, na_token)

    # observed-like: usamos dropna=True tras mapear NA→token para reducir la tabla
    tbl = pd.crosstab(x_, y_, dropna=True)
    if tbl.size == 0 or tbl.shape[0] < 2 or tbl.shape[1] < 2:
        return 0.0

    chi2, _, _, _ = chi2_contingency(tbl, correction=False)
    n = float(tbl.to_numpy().sum())
    if n <= 0.0:
        return 0.0

    phi2 = chi2 / n
    r, k = map(float, tbl.shape)
    n_minus_1 = max(1.0, n - 1.0)

    phi2corr = max(0.0, phi2 - ((k - 1.0) * (r - 1.0)) / n_minus_1)
    rcorr = r - ((r - 1.0) ** 2) / n_minus_1
    kcorr = k - ((k - 1.0) ** 2) / n_minus_1
    denom = max(1e-12, min(rcorr - 1.0, kcorr - 1.0))

    return float(np.sqrt(phi2corr / denom)) if denom > 0 else 0.0


@dataclass
class CramersVCorrelatedSelection:
    threshold: float = 0.8
    variables: list[str] | None = None
    selection_method: str = "cardinality"   # {"importance","cardinality","first"}
    importance_dict: dict[str, float] | None = None

    include_discrete: bool = True
    discrete_max_cardinality: int = 30
    # Limitar pares para evitar crosstabs gigantes
    max_pair_levels: int = 200_000
    na_token: str = "__NA__"
    # Muestrear filas para acelerar (0 = sin muestreo)
    row_sample: int = 100_000

    # Atributos tras fit()
    variables_: list[str] | None = None
    features_to_drop_: list[str] | None = None
    features_to_keep_: list[str] | None = None
    association_matrix_: pd.DataFrame | None = None

    # --------- helpers internos ---------
    @staticmethod
    def _to_name(v, cols: list[str]) -> str | None:
        """Mapea índices (int o str numérica) al nombre real; deja pasar nombres ya existentes."""
        # índice entero
        if isinstance(v, (int, np.integer)):
            vi = int(v)
            return cols[vi] if 0 <= vi < len(cols) else None
        # string
        if isinstance(v, str):
            if v in cols:
                return v
            if v.isdigit():
                vi = int(v)
                return cols[vi] if 0 <= vi < len(cols) else None
        # no soportado
        return None

    def _resolve_variables(self, X: pd.DataFrame) -> list[str]:
        cols = X.columns.astype(str).tolist()

        # Si el usuario pasó variables, normalizamos a NOMBRES
        if self.variables is not None:
            resolved: list[str] = []
            missing_like: list[object] = []

            for v in self.variables:
                name = self._to_name(v, cols)
                if name is None:
                    missing_like.append(v)
                else:
                    resolved.append(name)

            if missing_like:
                raise ValueError(f"Variables no encontradas / fuera de rango: {missing_like}")

            # de-duplicar preservando orden
            seen = set()
            resolved = [c for c in resolved if not (c in seen or seen.add(c))]
            return resolved

        # Auto-detección por tipo: categóricas + discretas (si procede)
        cats = X.select_dtypes(include=["object", "category", "string"]).columns.astype(str).tolist()

        if self.include_discrete:
            num_cols = X.select_dtypes(include=["number"]).columns.astype(str).tolist()
            discretes = [c for c in num_cols if X[c].nunique(dropna=False) <= self.discrete_max_cardinality]
            out, seen = [], set()
            for c in cats + discretes:
                if c not in seen:
                    out.append(c); seen.add(c)
            if not out:
                raise ValueError("No se encontraron variables categóricas/discretas adecuadas en X.")
            return out

        if not cats:
            raise ValueError("No se encontraron variables categóricas en X.")
        return cats

    # --------- API scikit ---------
    def fit(self, X: pd.DataFrame, y: pd.Series | None = None):
        X = X.copy()
        X.columns = X.columns.astype(str)

        # Muestreo opcional para acelerar
        if self.row_sample and len(X) > self.row_sample:
            X = X.sample(self.row_sample, random_state=42)

        self.variables_ = self._resolve_variables(X)
        idx = self.variables_

        # cardinalidades con NA-token (sin FutureWarning)
        nunique = {c: _prep_cat(X[c], self.na_token).nunique(dropna=False) for c in idx}

        # construir matriz de asociación
        assoc = pd.DataFrame(index=idx, columns=idx, dtype=float)
        for i, a in enumerate(idx):
            assoc.loc[a, a] = 1.0
            for b in idx[i + 1:]:
                # Saltar pares enormes
                if nunique[a] * nunique[b] > self.max_pair_levels:
                    assoc.loc[a, b] = assoc.loc[b, a] = np.nan
                    continue
                v = _cramers_v_bias_corrected(X[a], X[b], na_token=self.na_token)
                assoc.loc[a, b] = v
                assoc.loc[b, a] = v

        self.association_matrix_ = assoc

        # Estrategia de selección (tipo SmartCorrelatedSelection)
        to_drop: set[str] = set()
        kept: set[str] = set()

        if self.selection_method == "importance":
            if not self.importance_dict:
                raise ValueError("Debes pasar importance_dict cuando selection_method='importance'.")
            ordered = sorted(idx, key=lambda c: self.importance_dict.get(c, -np.inf), reverse=True)
        elif self.selection_method == "cardinality":
            ordered = sorted(idx, key=lambda c: (nunique[c], c))
        elif self.selection_method == "first":
            ordered = list(idx)
        else:
            raise ValueError(f"selection_method inválido: {self.selection_method}")

        for col in ordered:
            if col in to_drop:
                continue
            kept.add(col)
            partners = assoc.loc[col, :].dropna()
            high = partners[(partners.index != col) & (partners >= self.threshold)].index.tolist()
            for h in high:
                if h not in kept:
                    to_drop.add(h)

        self.features_to_keep_ = [c for c in ordered if c in kept]
        self.features_to_drop_ = [c for c in idx if c in to_drop]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.features_to_drop_ is None:
            raise RuntimeError("Debes llamar fit() antes de transform().")
        return X.drop(columns=self.features_to_drop_, errors="ignore")

    def fit_transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features: list[str] | None = None) -> list[str]:
        if self.features_to_keep_ is None:
            raise RuntimeError("Debes llamar fit() primero.")
        return self.features_to_keep_
