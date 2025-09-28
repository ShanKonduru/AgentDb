
import os
from dotenv import load_dotenv
import pytest

load_dotenv()
# openai_api_key = os.getenv("OPENAI_API_KEY")

def add(x, y):
    return x + y

def multiply(x, y):
    return x * y

def divide(x, y):
    return x / y

def subtract(x, y):
    return x - y

@pytest.mark.positive
def test_add_positive():
    assert add(2, 3) == 5

@pytest.mark.positive
def test_subtract_positive():
    assert subtract(10, 4) == 6

@pytest.mark.positive
def test_multiply_positive():
    assert multiply(5, 6) == 30

@pytest.mark.positive
def test_divide_positive():
    assert divide(10, 2) == 5

@pytest.mark.edge
def test_add_with_zero():
    assert add(5, 0) == 5

@pytest.mark.edge
def test_subtract_with_zero():
    assert subtract(10, 0) == 10

@pytest.mark.edge
def test_add_with_negative_numbers():
    assert add(-2, -3) == -5

@pytest.mark.edge
def test_multiply_by_zero():
    assert multiply(100, 0) == 0

@pytest.mark.edge
def test_divide_negative_numbers():
    assert divide(-10, -2) == 5
