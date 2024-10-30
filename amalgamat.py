print("\nPrzypadek 1\n")

x = 0.4 + 0.2
y = 0.6

if x == y:
    print("True")
else:
    print("False")

print("\nPrzypadek 2\n")

x = 0.4 + 0.3
y = 0.7

if x == y:
    print("True")
else:
    print("False")
    

print("\nJak sobie z tym poradzić?\n")

import math

x = 0.4 + 0.3
y = 0.7

if math.isclose(x, y, rel_tol=1e-9):
    print("True")
else:
    print("False")