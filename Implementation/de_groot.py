import numpy as np

epsilon = 0.00000000000001
limit = 1000

def generer_matrice(n):
    matrice = np.random.rand(n, n)
    matrice = matrice / matrice.sum(axis=1, keepdims=True)
    return matrice

def generer_opinions(n):
    return np.random.rand(n)

def phase(X,A):
    n=len(X)
    Y=[0 for _ in range(n)]
    for i in range(n-1):
        for j in range(n):
            Y[i+1]+=A[i][j]*X[j]
    return Y

def diff_vecteur(A,B):
    X=[]
    for i in range(len(A)):
        X.append(abs(A[i]-B[i]))
    return X

n=0
A=generer_matrice(5)
X=generer_opinions(5)
print("Les opinions initiaux étaient :", X)
while True:
    Y=X
    X=phase(Y,A)
    if max(diff_vecteur(X,Y)) < epsilon :
        convergence = X
        print("Les opinions finaux sont :",convergence)
        break
    n+=1
    if n == limit :
        print("Pas de convergence")


