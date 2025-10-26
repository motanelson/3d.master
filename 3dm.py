"""
einstein_field.py

Calcula G_{mu nu} = (8*pi*G / c^4) * T_{mu nu}.

Requisitos:
    pip install numpy sympy

Uso:
    python einstein_field.py
    -> executa exemplos numéricos e simbólicos
"""
import numpy as np
import math

# Opicional: para saída simbólica/analítica
try:
    import sympy as sp
    SYMBOLIC_AVAILABLE = True
except Exception:
    SYMBOLIC_AVAILABLE = False


# Valores padrão (SI)
G_NEWTON = 6.67430e-11         # m^3 kg^-1 s^-2
C_LIGHT = 299_792_458          # m / s

def coupling_constant(G=G_NEWTON, c=C_LIGHT):
    """Retorna k = 8*pi*G / c**4"""
    return 8.0 * math.pi * G / (c ** 4)

def validate_tensor(T):
    """Valida que T é um array 4x4 e retorna numpy.array(T)."""
    arr = np.array(T, dtype=float)
    if arr.shape != (4,4):
        raise ValueError(f"Tensor T deve ser shape (4,4). Shape dado = {arr.shape}")
    return arr

def is_symmetric(arr, tol=1e-12):
    return np.allclose(arr, arr.T, atol=tol, rtol=0)

def compute_G_from_T(T, G=G_NEWTON, c=C_LIGHT):
    """
    Calcula G_{mu nu} = k * T_{mu nu}.
    T: array-like 4x4 (componente T_{mu nu} em unidades SI: J/m^3 etc)
    Retorna numpy array 4x4.
    """
    arr = validate_tensor(T)
    k = coupling_constant(G, c)
    return k * arr

def compute_T_from_G(G_tensor, G=G_NEWTON, c=C_LIGHT):
    """Inverte a relação: T = G_tensor / k"""
    arr = validate_tensor(G_tensor)
    k = coupling_constant(G, c)
    return arr / k

def pretty_print_tensor(arr, name="Tensor"):
    """Imprime o tensor 4x4 com formatação."""
    arr = np.array(arr, dtype=float)
    print(f"\n{name} (shape {arr.shape}):")
    for row in arr:
        print("  [" + ", ".join(f"{v: .6e}" for v in row) + "]")


# -----------------------
# Exemplos / demo
# -----------------------
def demo_numeric():
    print("DEMO NUMÉRICA — cálculo direto de G_{μν} = k * T_{μν}")
    # Exemplo: tensor diagonal (ex.: gás perfeito no referencial local)
    # T^μ_ν diag: T00 = rho c^2, T11 = p, T22 = p, T33 = p (mas aqui fornecemos T_{μν} diretamente)
    rho = 1.0e5           # massa densidade kg/m^3 (exemplo)
    p = 1.0e9             # pressão em Pa (exemplo)
    # montar T_{mu nu} em unidades SI (simplificação: usar diag)
    T = np.zeros((4,4))
    T[0,0] = rho * (C_LIGHT**2)   # energia densidade (J/m^3 ~ kg m^-1 s^-2)
    T[1,1] = p
    T[2,2] = p
    T[3,3] = p

    pretty_print_tensor(T, "T_{μν} (input)")

    k = coupling_constant()
    print(f"\nConstante k = 8πG / c^4 = {k:.6e} (SI units: 1/(Pa) roughly)")

    G_tensor = compute_G_from_T(T)
    pretty_print_tensor(G_tensor, "G_{μν} (resultado)")

    # Valida simetria
    print("\nValidação de simetria de T:", is_symmetric(T))
    print("Validação de simetria de G:", is_symmetric(G_tensor))

def demo_symbolic():
    if not SYMBOLIC_AVAILABLE:
        print("\nSympy não está disponível — instala com: pip install sympy para demo simbólica.")
        return
    print("\nDEMO SIMBÓLICA — expressão simbólica de G_{μν} em termos de T_{μν}, G, c.")
    Gsym, csym, pi = sp.symbols('G c pi')
    k_sym = 8 * pi * Gsym / csym**4

    # Tensores simbólicos: escrever só a componente (0,0) como exemplo
    T00 = sp.symbols('T00')
    G00 = sp.simplify(k_sym * T00)
    print("G_00 =", sp.pretty(G00))
    # Exemplo de matriz simbólica (útil para imprimir a relação)
    T_mat = sp.Matrix([[sp.symbols(f"T{i}{j}") for j in range(4)] for i in range(4)])
    G_mat = sp.simplify(k_sym * T_mat)
    print("\nG_{μν} = k * T_{μν}  com k = 8πG/c^4  (mostrando as 4x4 simbólicas):")
    sp.pprint(G_mat)

def main():
    print("=== Einstein-field quick calculator ===")
    demo_numeric()
    demo_symbolic()
    print("\nExemplo: para usar as funções em outro ficheiro, importa compute_G_from_T/compute_T_from_G.")
    print("Nota: esta aplicação NÃO resolve as equações diferenciais (valores do tensor de curvatura) —")
    print("aplica apenas a relação algébrica G_{μν} = k T_{μν} assumindo conheces T_{μν}.")
print("\033c\033[43;30m\ndata\n")

if __name__ == "__main__":
    main()

