x = "1.5"
print(type(x))

try:
    while int(x) + 1:
        print("A")
except ValueError:
    print("B")
except TypeError:
    print("C")
except KeyError:
    print("D")