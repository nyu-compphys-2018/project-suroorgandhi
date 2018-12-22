import numpy as np
import math
import matplotlib.pyplot as plt

# N = u^2*f(u), so Ndot = partial_N/partial_t is now fdot (partial_f/partial_t) with...
# ... f = N/u^2
def find_Ndot(N, G, u, dlnTdt_k, get_dNdu):
    # G (Gamma) is an NxN matrix
    n = len(N) ;
    Ndot = np.zeros(n) ; #to store partial N/partial t
    dNdu = get_dNdu(N, u) ;
    delu = (u[-1]-u[0]) / n ;
    for i in range(1, n-1):
        for j in range(1, n-1):
            Ndot[i] += 4*np.pi*delu* ( G[j][i] * N[j] * u[i]**2 - \
                                        G[i][j] * N[i] * u[j]**2 ) ;
        Ndot[i] += 0.5 * dlnTdt_k * ( u[i] * dNdu[i] + N[i]) ;
    return Ndot ;

def N_initial(u):
    # return \
    # 4*np.pi * u**2 * ( 1.0 / (2.*np.pi) )**(3./2.) * np.exp( - u**2 / 2. )
    return 1./(2.*np.pi) *(1 + np.sin(u-np.pi/2.)) ;
    # return 1./(2*np.pi)*np.ones(len(u)) ;

def N_MB(u):
    return \
    4*np.pi * u**2 * ( 1.0 / (2.*np.pi) )**(1.5) * np.exp( - u**2 / 2. )

def RK4(h, N, find_Ndot, *args):
    n = len(N) ;
    dNdt = np.zeros(n);
    k1 = h*find_Ndot(N, G, u, dlnTdt_k, get_dNdu) ;
    k2 = h*find_Ndot(N + 0.5*k1, G, u, dlnTdt_k, get_dNdu) ;
    k3 = h*find_Ndot(N + 0.5*k2, G, u, dlnTdt_k, get_dNdu) ;
    k4 = h*find_Ndot(N + k3, G, u, dlnTdt_k, get_dNdu) ;

    dNdt = (k1 + 2*k2 + 2*k3 + k4) / 6 ; # N += (k1)
    return dNdt ;

def get_dNdu(N,u): #centered difference
    n = len(N) ;
    dNdu = np.zeros(n)
    # dfdu[0] = (f[0] - f[1])/(u[0] - u[1])
    # dfdu[0] = 0 ;

    for i in range(1,n-1):
        dNdu[i] = (N[i+1] - N[i-1])/(u[i+1]-u[i-1])

    # dfdu[-1] = (f[-1] - f[-2])/(u[-1] - u[-2])
    # dfdu[-1] = 0 ;
    return dNdu ;

def Gamma(u, G0):
    G = np.empty([len(u), len(u)]) ;
    for i in range(len(u)) :
        for j in range(len(u)):
            G[i][j] = G0 * np.exp(- 0.5 * u[j]**2) ; #satisfies detailed balance
    return G ;

def get_T(a, b, t, h) : #time-varying temperature, T
    T = np.ones( int((tf-t0)/h) ) ;
    # a = 150 ;
    # b = 20 ;
    # T = np.pi/2 + np.arctan(a * t - b) ;
    return T ;

def get_dlnTdt(a, b, t, h): #analytical derivative
     # dlnTdt = (2 * a) / ( (1 + (b-a*t)**2) * ( np.pi - 2*np.arctan(b - a*t)) ) ;
     dlnTdt = np.zeros(len(t)) ;
     # dlnTdt = -.1*(t-0.02) * np.exp(-((t-0.02)/0.01)**2) ;
     # for i in range(len(t)) :
        # dlnTdt[i] = 2 * np.exp(-(b-t[i])**2) / ( np.sqrt(np.pi) * (a - math.erf(b-t[i]) ) ) ;
     return dlnTdt ;

if __name__=='__main__':
    n = int(input("Input n, the dimension of nxn matrix G:"))
    t0 = 0.0 ;
    tf = 5. ;
    h = 0.005 ;
    t = np.arange(t0,tf,h) ;
    #initiate matrices and vectors
    G0 = 1.0 ;
    # G = np.ones([n,n])*2.1 ;    #this needs to be determined
    u = np.linspace(0, 2*np.pi, n) ;
    # "   "   ""
    # defining u-array the way x is defined in dm_baryon_fp.py
    # "   "   ""
    # umin = 1e-4 ;
    # umax = 10. ;
    # dlnu = 1e-2 ;
    # Nu   = int(np.log(umax/umin)/dlnu) + 1
    # utab = umin *exp(dlnu *np.array([i for i in range(Nu)]))
    # "   "   ""
    # "   "   ""
    G  = Gamma(u, G0)

    N_MB = N_MB(u) ;
    # N = N_initial(u) ;
    N = N_MB ;
    Nsol = np.zeros( [len(t), n] ) ; # array to store solutions
    Nsol[0] = N ;
    plt.plot(u, N_MB, label = 'N_MB')
    # plt.legend()
    # plt.show()

    plt.plot(u, Nsol[0], label='t = 0.0')
    plt.title('Nsol')
    plt.legend()

    plt.show()

    plt.plot(u, Nsol[0], ls = ':', color = 'k', label='t = 0.0')
    a = 5. ;
    b = 15. ;
    dlnTdt = get_dlnTdt(a, b, t, h) ;
    for k in range(1, len(t)) :
        dlnTdt_k = dlnTdt[k] ;
        Nsol[k][1:-1] = N[1:-1] + \
                        RK4(h, N, find_Ndot, N, G, u, dlnTdt_k, get_dNdu)[1:-1] ;
        N = Nsol[k]
        # "   "   ""
        # trying redshifting N
        # "   "   ""
        # # N[:Nu-1] = N[1:]
        # # N[Nu-1]  = 0.
        # "   "   ""
        # "   "   ""
        N[0] = 0 ;
        N[-1] = 0 ;

        if k%100 == 0:
            if len(t)-1 - k == 100 :
                plt.scatter(u, Nsol[k], marker = '*', c= 'blue', \
                                label='end t = %(x).4f' %{ 'x':h*k})
                plt.xlabel('u')
                plt.ylabel('N(u)')
            else :
                plt.plot(u, Nsol[k], label='t = %(x).4f' %{ 'x':h*k})
                plt.xlabel('u')
                plt.ylabel('N(u)')
    plt.scatter(u, N_MB, marker = 'x',c = 'k' ,label='N_MB')

    plt.legend()
    plt.show()
    plt.plot(t, get_dlnTdt(a, b, t, h), label = 'dlnT/dt')
    plt.plot(t, get_T(a, b, t, h), label = 'T(t)')
    plt.xlabel('t')
    plt.legend()
    plt.show()
