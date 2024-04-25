import importlib
import subprocess

# Lista pakietów do sprawdzenia i ich nazwy w pip
packages = {
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "subprocess":'subprocess',
    "importlib":'importlib',
    'sklearn':'sklearn',
    'logging':'logging',
    'requests':'requests',
    're':'re',
    'time':'time',
    'pathlib':'pathlib',
    'pandas':'pandas',
    'numpy':'numpy'
}

def check_and_install_packages(packages):
    for module_name, pip_name in packages.items():
        try:
            importlib.import_module(module_name)
            print(f"Moduł {module_name} jest już zainstalowany.")
        except ImportError:
            if pip_name is not None:
                print(f"Instalowanie {module_name}...")
                subprocess.run(["pip", "install", pip_name], check=True)
            else:
                print(f"Nie znaleziono pakietu {module_name} w PyPI, sprawdź instalację ręcznie.")


check_and_install_packages(packages)
