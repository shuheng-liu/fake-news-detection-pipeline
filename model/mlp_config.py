sizes1 = [(200 * (n + 1),) for n in range(4)]
sizes2 = [(200 * (n + 1), 200 * (n + 1)) for n in range(4)]
sizes3 = [(200 * (n + 1), 100 * (n + 1)) for n in range(4)]
sizes = sizes1 + sizes2 + sizes3

random_seed = 0

if __name__ == '__main__':
    variables = vars().copy()
    print("MLP configurations: ")
    for name in variables:
        if name.startswith("__") or name.endswith("__"):
            continue
        print("\t{}: {}\n\t {}".format(name,type(variables[name]), variables[name]))
