import numpy as np
from app import y_pred, y_test, model_accuracy

def test_predictions_not_none():
    """
    Test 1: Sprawdza, czy otrzymujemy jakąkolwiek predykcję.
    """
    assert y_pred is not None, "Predykcje nie powinny być None."

def test_predictions_length():
    """
    Test 2: Sprawdza, czy długość listy predykcji jest większa od 0 i czy odpowiada przewidywanej liczbie próbek testowych.
    """
    assert len(y_pred) > 0, "Predykcje nie mogą być puste."
    assert len(y_pred) == len(y_test), "Liczba predykcji musi odpowiadać liczbie próbek testowych."

def test_predictions_value_range():
    """
    Test 3: Sprawdza, czy wartości w predykcjach mieszczą się w spodziewanym zakresie.
    """
    for p in y_pred:
        assert p in [0, 1], f"Predykcja {p} jest poza dozwolonym zakresem klas (0, 1)."

def test_model_accuracy():
    """
    Test 4: Sprawdza, czy model osiąga co najmniej 70% dokładności.
    """
    assert model_accuracy >= 0.70, f"Dokładność modelu ({model_accuracy * 100}%) jest mniejsza niż wymagane 70%."