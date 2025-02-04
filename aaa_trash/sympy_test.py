import sympy as sp

y = sp.symbols('y')
x = sp.symbols('x')

f = x**2 + 2*y + 1

f_prime = f.diff(x) + f.diff(y)

print(f_prime)