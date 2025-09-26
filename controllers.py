import os


DATASETS = {
    "1": ("kecil.csv", "Small dataset selected: kecil.csv"),
    "2": ("sedang.csv", "Medium dataset selected: sedang.csv"),
    "3": ("besar.csv", "Big dataset selected: besar.csv"),
}


def ask_dataset_selection() -> str:
    while True:
        print("Choose dataset:")
        print("1) Small (kecil.csv)")
        print("2) Medium (sedang.csv)")
        print("3) Big (besar.csv)")
        choice = input("Enter 1, 2, or 3: ").strip()
        if choice in DATASETS:
            path, msg = DATASETS[choice]
            print(msg)
            if not os.path.isfile(path):
                print(f"Warning: '{path}' not found. Will attempt to fall back if supported.")
            return path
        else:
            print("Invalid input. Please enter 1, 2, or 3.")


def ask_logging_selection() -> bool:
    while True:
        print("Logging mode:")
        print("1) Log all details to file (no logs in terminal)")
        print("2) Show only conclusion in terminal (no file log)")
        choice = input("Enter 1 or 2: ").strip()
        if choice == "1":
            return True
        if choice == "2":
            return False
        print("Invalid input. Please enter 1 or 2.")


def ask_parameter_selection(default_params: dict, a_const: float, epsilon_const: float) -> dict:
    while True:
        print("Parameters:")
        print("1) Use default parameters")
        print("2) Enter custom parameters")
        choice = input("Enter 1 or 2: ").strip()
        if choice == "1":
            params = default_params.copy()
            params["A"] = a_const
            params["epsilon"] = epsilon_const
            return params
        if choice == "2":
            while True:
                try:
                    n_sf = int(input("Enter number of sailfish: ").strip())
                    n_sd = int(input("Enter number of sardines: ").strip())
                    max_iter = int(input("Enter max iterations: ").strip())
                except ValueError:
                    print("Invalid number. Please enter integers.")
                    continue
                if n_sd <= n_sf:
                    print("Error: sardine population must be greater than sailfish population.")
                    continue
                if max_iter <= 0:
                    print("Error: max iterations must be greater than 0.")
                    continue
                return {
                    "n_sailfish": n_sf,
                    "n_sardines": n_sd,
                    "max_iter": max_iter,
                    "A": a_const,
                    "epsilon": epsilon_const,
                }
        print("Invalid input. Please enter 1 or 2.")




