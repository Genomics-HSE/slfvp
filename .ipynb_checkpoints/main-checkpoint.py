from slfvp import *

a = Model(
    rho = 1000,
    L = 1,
    lamda = 1,
    u0 = 0.4,
    alpha = 1,
    theta = 0.3,
    n_alleles = 10
)

a.generate_dynamic(1000)

a.initiate(0.4)

a.run()