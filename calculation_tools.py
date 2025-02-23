from langchain_core.tools import tool
import json


@tool
def multiply(a: int, b: int) -> str:
    """Multiplies two numbers."""
    return str(a * b)

@tool
def add(a: int, b: int) -> str:
    """Adds two numbers."""
    return str(a + b)

@tool
def calculate_based_on_income(incomes:list[dict[str, str | float]], amount:float) -> str:
    """Splits the given amount across incomes based on each income's percentage of the total income.
    Args:
        incomes: list of dictionaries with 'name' and 'salary' keys
        amount: float
    """

    total_salary = sum(income['salary'] for income in incomes)
    
    # Avoid division by zero if total salary is zero
    if total_salary == 0:
        return [{'name': income['name'], 'amount': 0} for income in incomes]
    
    result = []
    
    for income in incomes:
        allocated_amount = (income['salary'] / total_salary) * amount
        result.append({'name': income['name'], 'amount': allocated_amount})
    
    return json.dumps(result)

@tool
def calculate_equally(number_of_people:int, amount:float) -> str:
    """Splits the given amount equallt beween number of people.

    Args:
        number_of_people: int
        amount: float
    """
    
    return str(amount/number_of_people) 
