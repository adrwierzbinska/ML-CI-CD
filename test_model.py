import numpy as np
from model import train_and_predict, get_accuracy

def test_predictions_not_none():
    """
    Test 1: Sprawdza, czy otrzymujemy jakąkolwiek predykcję.
    """
    preds, _ = train_and_predict()
    assert preds is not None, "Predykcje nie powinny być None."

def test_predictions_length():
    """
    Test 2 (na maksymalną ocenę 5): Sprawdza, czy długość listy predykcji jest większa od 0 i czy odpowiada przewidywanej liczbie próbek testowych.
    """
    preds, y_test = train_and_predict()
    assert len(preds) > 0, "Predykcje nie mogą być puste."
    assert len(preds) == len(y_test), "Liczba predykcji musi odpowiadać liczbie próbek testowych."

def test_predictions_value_range():
    """
    Test 3 (na maksymalną ocenę 5): Sprawdza, czy wartości w predykcjach mieszczą się w spodziewanym zakresie: Dla zbioru Iris mamy 3 klasy (0, 1, 2).
    """
    preds, _ = train_and_predict()
    for p in preds:
        assert p in [0, 1, 2], f"Predykcja {p} jest poza dozwolonym zakresem klas (0, 1, 2)."

def test_model_accuracy():
    """
    Test 4 (na maksymalną ocenę 5): Sprawdza, czy model osiąga co najmniej 70% dokładności (przykładowy warunek, można dostosować do potrzeb).
    """
    accuracy = get_accuracy()
    assert accuracy >= 0.70, f"Dokładność modelu ({accuracy * 100}%) jest mniejsza niż wymagane 70%."