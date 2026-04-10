# Trabalho-da-faculdade-Cefet
Código para estudar frequências naturais 
import numpy as np
import matplotlib.pyplot as plt

print("\n===== ANÁLISE DE EIXO COM ROTORES =====\n")

g = 10

# =============================
# ENTRADA DE DADOS
# =============================

E = float(input("Módulo de elasticidade E (GPa): ")) * 1e9
Sy = float(input("Limite de escoamento (MPa): ")) * 1e6
d = float(input("Diâmetro do eixo (mm): ")) / 1000
L = float(input("Comprimento do eixo (m): "))

n = int(input("Número de rotores: "))

pos = []
mass = []
P = []

for i in range(n):

    print(f"\nRotor {i+1}")

    x = float(input("Posição (m): "))
    m = float(input("Massa (kg): "))

    pos.append(x)
    mass.append(m)

    P.append(m*g)

pos = np.array(pos)
mass = np.array(mass)
P = np.array(P)

# =============================
# PROPRIEDADES DA SEÇÃO
# =============================

I = np.pi*d**4/64
c = d/2

# =============================
# REAÇÕES NOS APOIOS
# =============================

sumM = np.sum(P*pos)

RB = sumM/L
RA = np.sum(P) - RB

print("\nREAÇÕES")
print("RA =",RA,"N")
print("RB =",RB,"N")

# =============================
# DISCRETIZAÇÃO DO EIXO
# =============================

x = np.linspace(0,L,1000)

V = np.zeros(len(x))
M = np.zeros(len(x))

for i in range(len(x)):

    V[i] = RA

    for j in range(n):

        if x[i] >= pos[j]:
            V[i] -= P[j]

for i in range(1,len(x)):

    dx = x[i]-x[i-1]
    M[i] = M[i-1] + V[i]*dx

# =============================
# MOMENTO MÁXIMO
# =============================

Mmax = np.max(np.abs(M))
xM = x[np.argmax(np.abs(M))]

print("\nMOMENTO MÁXIMO")
print("Mmax =",Mmax,"N.m")
print("Posição =",xM,"m")

# =============================
# TENSÃO DE FLEXÃO
# =============================

sigma = M*c/I
sigma_max = np.max(np.abs(sigma))
x_sigma = x[np.argmax(np.abs(sigma))]

print("\nTENSÃO MÁXIMA DE FLEXÃO")
print("σ =",sigma_max/1e6,"MPa")
print("Posição =",x_sigma,"m")

# =============================
# TENSÕES PRINCIPAIS
# =============================

sigma1 = sigma_max
sigma2 = 0

print("\nTENSÕES PRINCIPAIS")
print("σ1 =",sigma1/1e6,"MPa")
print("σ2 =",sigma2)

# =============================
# VON MISES
# =============================

sigma_vm = abs(sigma1)

print("\nTENSÃO VON MISES")
print("σ_vm =",sigma_vm/1e6,"MPa")

# =============================
# CRITÉRIOS DE FALHA
# =============================

print("\nCRITÉRIO DE FALHA")

if sigma_vm > Sy:
    print("Von Mises: FALHA")
else:
    print("Von Mises: NÃO FALHA")

if sigma1 > Sy/2:
    print("Tresca: FALHA")
else:
    print("Tresca: NÃO FALHA")

# =============================
# LINHA ELÁSTICA
# =============================

theta = np.zeros(len(x))
y = np.zeros(len(x))

for i in range(1,len(x)):

    dx = x[i]-x[i-1]

    theta[i] = theta[i-1] + M[i]/(E*I)*dx
    y[i] = y[i-1] + theta[i]*dx

# condição de contorno
y = y - (x/L)*y[-1]

ymax = np.max(np.abs(y))
xymax = x[np.argmax(np.abs(y))]

print("\nDEFLEXÃO MÁXIMA")
print("ymax =",ymax*1000,"mm")
print("Posição =",xymax,"m")

print("\nDEFLEXÃO NOS ROTORES")

defl = []

for i in range(n):

    dtemp = np.interp(pos[i],x,y)

    defl.append(dtemp)

    print("Rotor",i+1,"=",dtemp*1000,"mm")

defl = np.array(defl)

# =============================
# COEFICIENTES DE INFLUÊNCIA
# =============================

alpha = np.zeros((n,n))

for j in range(n):

    F = np.zeros(n)
    F[j] = 1.0   # força unitária

    sumM = np.sum(F*pos)

    RB = sumM/L
    RA = np.sum(F) - RB

    Mtemp = np.zeros(len(x))

    for i in range(len(x)):

        Mtemp[i] = RA*x[i]

        for k in range(n):
            if x[i] >= pos[k]:
                Mtemp[i] -= F[k]*(x[i]-pos[k])

    thetat = np.zeros(len(x))
    yt = np.zeros(len(x))

    for i in range(1,len(x)):

        dx = x[i]-x[i-1]

        thetat[i] = thetat[i-1] + Mtemp[i]/(E*I)*dx
        yt[i] = yt[i-1] + thetat[i]*dx

    yt = yt - (x/L)*yt[-1]

    for i in range(n):

        alpha[i,j] = np.interp(pos[i],x,yt)

# FORÇA SIMETRIA (MAXWELL-BETTI)
alpha = (alpha + alpha.T)/2

print("\nCOEFICIENTES DE INFLUÊNCIA")
print(alpha)

# =============================
# FREQUÊNCIA RAYLEIGH
# =============================

num = 0
den = 0

for i in range(n):

    num += mass[i]*g*abs(defl[i])
    den += mass[i]*(defl[i]**2)

w = np.sqrt(num/den)
f_ray = w/(2*np.pi)

print("\nFREQUÊNCIA NATURAL (RAYLEIGH)")
print("f =",f_ray,"Hz")

# =============================
# DUNKERLEY
# =============================

sum_inv = 0

for i in range(n):

    wi = np.sqrt(g/abs(defl[i]))

    sum_inv += 1/(wi**2)

wd = np.sqrt(1/sum_inv)

f_d = wd/(2*np.pi)

print("\nFREQUÊNCIA NATURAL (DUNKERLEY)")
print("f =",f_d,"Hz")

# =============================
# SOLUÇÃO EXATA
# =============================

alpha = np.abs(alpha)

K = np.linalg.inv(alpha)

Mmat = np.diag(mass)

A = np.linalg.inv(Mmat) @ K

eigvals , eigvecs = np.linalg.eig(A)

eigvals = np.abs(eigvals)

w = np.sqrt(eigvals)

freq = w/(2*np.pi)

freq = np.sort(freq)

print("\nFREQUÊNCIAS NATURAIS (SOLUÇÃO EXATA)")
print("f1 =",freq[0],"Hz")
print("f2 =",freq[1],"Hz")

# =============================
# GRÁFICOS
# =============================

plt.figure()
plt.plot(x,V)
plt.title("Diagrama de Esforço Cortante")
plt.xlabel("x (m)")
plt.ylabel("V (N)")
plt.grid()

plt.figure()
plt.plot(x,M)
plt.title("Diagrama de Momento Fletor")
plt.xlabel("x (m)")
plt.ylabel("M (N.m)")
plt.grid()

plt.figure()
plt.plot(x,y*1000)
plt.title("Deflexão do eixo")
plt.xlabel("x (m)")
plt.ylabel("Deflexão (mm)")
plt.grid()

plt.show()
