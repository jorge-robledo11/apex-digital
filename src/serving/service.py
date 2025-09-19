import numpy as np
import pandas as pd
import time
from .registry import ModelBundle
from .schemas import PredictionResult

class ModelService:
    def __init__(self, bundle: ModelBundle):
        self.bundle = bundle
        
    @property
    def class_names(self) -> list[str]:
        return self.bundle.class_names or ["DIGITAL", "TELEFONO", "VENDEDOR"]
    
    @property
    def input_cols(self) -> list[str] | None:
        return self.bundle.input_cols
    
    @property  
    def categorical_cols(self) -> list[str]:
        return self.bundle.categorical_cols
    
    def predict_proba(self, df: pd.DataFrame) -> tuple[np.ndarray, float]:
        """Predicción con múltiples estrategias para bypass de signature"""
        start_time = time.time()
        
        print(f"🔍 Predicción - DataFrame input:")
        print(f"   Shape: {df.shape}")
        print(f"   Categorical count: {len([col for col in df.columns if df[col].dtype.name == 'category'])}")
        
        # ✅ ESTRATEGIA 1: Acceder al XGBoost real a través de múltiples capas
        try:
            pyfunc_model = self.bundle.pyfunc_model
            print(f"🎯 pyfunc_model type: {type(pyfunc_model)}")
            
            # Buscar el modelo XGBoost real en diferentes atributos
            xgb_model = None
            
            # Opción 1: _model_impl
            if hasattr(pyfunc_model, '_model_impl'):
                model_impl = pyfunc_model._model_impl
                print(f"🔍 _model_impl type: {type(model_impl)}")
                
                # Buscar más profundo
                if hasattr(model_impl, 'xgb_model'):
                    xgb_model = model_impl.xgb_model
                    print(f"✅ xgb_model encontrado: {type(xgb_model)}")
                elif hasattr(model_impl, '_model'):
                    xgb_model = model_impl._model
                    print(f"✅ _model encontrado: {type(xgb_model)}")
                elif hasattr(model_impl, 'model'):
                    xgb_model = model_impl.model
                    print(f"✅ model encontrado: {type(xgb_model)}")
            
            # Opción 2: Directamente en pyfunc
            if xgb_model is None:
                for attr in ['xgb_model', '_model', 'model']:
                    if hasattr(pyfunc_model, attr):
                        xgb_model = getattr(pyfunc_model, attr)
                        print(f"✅ {attr} encontrado directamente: {type(xgb_model)}")
                        break
            
            # Intentar predicción directa
            if xgb_model is not None:
                print(f"🚀 Intentando predicción directa con: {type(xgb_model)}")
                try:
                    probabilities = xgb_model.predict_proba(df)
                    processing_time = (time.time() - start_time) * 1000
                    print(f"✅ Predicción exitosa: shape={probabilities.shape}")
                    return probabilities, processing_time
                except Exception as e:
                    print(f"❌ Predicción directa falló: {e}")
            
        except Exception as e:
            print(f"⚠️ Error en estrategia 1: {e}")
        
        # ✅ ESTRATEGIA 2: Usar pyfunc pero sin signature enforcement
        try:
            print("🎯 Estrategia 2: Bypass signature enforcement")
            
            # Temporalmente deshabilitar signature checking
            original_predict = self.bundle.pyfunc_model.predict
            
            # Acceder al modelo real y usar predict directamente
            if hasattr(self.bundle.pyfunc_model, '_model_impl'):
                model_impl = self.bundle.pyfunc_model._model_impl
                
                # Forzar predict_proba sin validación
                if hasattr(model_impl, 'predict'):
                    probabilities = model_impl.predict(df)
                    processing_time = (time.time() - start_time) * 1000
                    print(f"✅ Estrategia 2 exitosa: shape={probabilities.shape}")
                    return probabilities, processing_time
                    
        except Exception as e:
            print(f"⚠️ Error en estrategia 2: {e}")
        
        # ✅ ESTRATEGIA 3: Crear XGBoost DMatrix manualmente
        try:
            print("🎯 Estrategia 3: DMatrix manual")
            import xgboost as xgb
            
            # Crear DMatrix con enable_categorical=True
            dmatrix = xgb.DMatrix(df, enable_categorical=True)
            
            # Buscar el booster real
            if hasattr(self.bundle.pyfunc_model, '_model_impl'):
                model_impl = self.bundle.pyfunc_model._model_impl
                if hasattr(model_impl, 'get_booster'):
                    booster = model_impl.get_booster()
                    probabilities = booster.predict(dmatrix)
                    processing_time = (time.time() - start_time) * 1000
                    print(f"✅ Estrategia 3 exitosa: shape={probabilities.shape}")
                    return probabilities, processing_time
                    
        except Exception as e:
            print(f"⚠️ Error en estrategia 3: {e}")
        
        # ✅ FALLBACK FINAL: Error controlado
        print("❌ Todas las estrategias fallaron")
        raise RuntimeError(
            "No se pudo realizar predicción. "
            "Problema de compatibilidad entre modelo entrenado (categorical) y signature (string). "
            "Considerar reentrenar modelo con signature correcta."
        )
    
    def predict_labels(self, proba: np.ndarray) -> list[PredictionResult]:
        """Convierte probabilidades a resultados estructurados"""
        predictions = []
        
        for prob_row in proba:
            max_idx = np.argmax(prob_row)
            predicted_class = self.class_names[max_idx]
            confidence = float(prob_row[max_idx])
            
            result = PredictionResult(
                probabilities=[float(p) for p in prob_row],
                predicted_class=predicted_class,
                confidence=confidence
            )
            predictions.append(result)
        
        return predictions
